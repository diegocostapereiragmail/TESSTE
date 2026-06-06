#!/usr/bin/env python3
"""
Zabbix Unsupported Items Diagnostic Agent with Google Gemini AI Analysis.

Requirements: requests (pip install requests)
Environment variables required:
    ZABBIX_URL       - Zabbix base URL (e.g. https://zabbix.empresa.com.br)
    ZABBIX_USER      - Zabbix admin username
    ZABBIX_PASSWORD  - Zabbix admin password
    GEMINI_API_KEY   - Google Gemini API key (free at aistudio.google.com)
"""

import os
import sys
import json
import time
import hashlib
from datetime import datetime, timezone
from pathlib import Path

try:
    import requests
except ImportError:
    print("ERRO: Biblioteca 'requests' não encontrada. Execute: pip install requests")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

GEMINI_ENDPOINT = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "gemini-2.5-flash:generateContent"
)
CACHE_FILE = Path(".zabbix_gemini_cache.json")
REPORT_FILE = Path("zabbix_unsupported_report.md")
MAX_ITEMS_PER_GEMINI_CALL = 20
MAX_GEMINI_CALLS = 10
UNSUPPORTED_OLD_DAYS = 7


# ---------------------------------------------------------------------------
# Environment validation
# ---------------------------------------------------------------------------

def load_env() -> dict:
    required = {
        "ZABBIX_URL": os.environ.get("ZABBIX_URL", "").rstrip("/"),
        "ZABBIX_USER": os.environ.get("ZABBIX_USER", ""),
        "ZABBIX_PASSWORD": os.environ.get("ZABBIX_PASSWORD", ""),
        "GEMINI_API_KEY": os.environ.get("GEMINI_API_KEY", ""),
    }
    missing = [k for k, v in required.items() if not v]
    if missing:
        print("=" * 60)
        print("ERRO: Variáveis de ambiente obrigatórias não definidas:")
        for var in missing:
            print(f"  → {var}")
        print()
        print("Configure as variáveis antes de executar:")
        print('  export ZABBIX_URL="https://zabbix.sua-empresa.com.br"')
        print('  export ZABBIX_USER="Admin"')
        print('  export ZABBIX_PASSWORD="sua_senha"')
        print('  export GEMINI_API_KEY="sua_chave_gemini"')
        print()
        print("Chave Gemini gratuita em: https://aistudio.google.com/")
        print("=" * 60)
        sys.exit(1)
    return required


# ---------------------------------------------------------------------------
# HTTP helpers with retry + exponential backoff
# ---------------------------------------------------------------------------

def http_post(url: str, payload: dict, headers: dict = None, max_retries: int = 3) -> dict:
    headers = headers or {"Content-Type": "application/json"}
    for attempt in range(max_retries):
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            wait = 2 ** attempt
            if attempt < max_retries - 1:
                print(f"  [retry {attempt + 1}/{max_retries}] Erro: {exc}. Aguardando {wait}s...")
                time.sleep(wait)
            else:
                raise


# ---------------------------------------------------------------------------
# Zabbix API
# ---------------------------------------------------------------------------

class ZabbixClient:
    def __init__(self, url: str, user: str, password: str):
        self.url = f"{url}/api_jsonrpc.php"
        self.user = user
        self.password = password
        self.auth_token: str = ""
        self._req_id = 1

    def _call(self, method: str, params: dict) -> dict:
        payload = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
            "id": self._req_id,
        }
        if self.auth_token:
            payload["auth"] = self.auth_token
        self._req_id += 1
        data = http_post(self.url, payload)
        if "error" in data:
            raise RuntimeError(f"Zabbix API error [{method}]: {data['error']}")
        return data.get("result", {})

    def login(self):
        print("→ Autenticando no Zabbix...")
        result = self._call("user.login", {
            "username": self.user,
            "password": self.password,
        })
        self.auth_token = result
        print(f"  ✓ Autenticado (token: {result[:8]}...)")

    def get_unsupported_items(self) -> list:
        print("→ Buscando itens não suportados (state=1)...")
        items = self._call("item.get", {
            "output": ["itemid", "name", "key_", "error", "lastclock", "hostid", "status", "state"],
            "filter": {"state": 1},
            "selectHosts": ["hostid", "host", "name", "status"],
            "limit": 500,
        })
        print(f"  ✓ {len(items)} itens não suportados encontrados.")
        return items

    def get_triggers_for_items(self, item_ids: list) -> list:
        if not item_ids:
            return []
        print(f"→ Buscando triggers para {len(item_ids)} item(s)...")
        triggers = self._call("trigger.get", {
            "output": ["triggerid", "description", "priority", "value", "lastchange"],
            "itemids": item_ids,
            "selectItems": ["itemid", "name"],
            "filter": {"status": 0},
            "expandDescription": True,
        })
        print(f"  ✓ {len(triggers)} triggers encontradas.")
        return triggers


# ---------------------------------------------------------------------------
# Cache management
# ---------------------------------------------------------------------------

def load_cache() -> dict:
    if CACHE_FILE.exists():
        try:
            return json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def save_cache(cache: dict):
    CACHE_FILE.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


def cache_key(items_data: list) -> str:
    serialized = json.dumps(items_data, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(serialized.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Gemini API
# ---------------------------------------------------------------------------

class GeminiClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.calls_made = 0
        self.cache = load_cache()

    def analyze(self, items_data: list, triggers_data: list) -> str:
        key = cache_key(items_data)
        if key in self.cache:
            print(f"  ↩ Cache hit para grupo de erros (key={key})")
            return self.cache[key]

        if self.calls_made >= MAX_GEMINI_CALLS:
            return "(limite de chamadas Gemini atingido — análise não disponível)"

        prompt = (
            "Você é um especialista em monitoramento Zabbix e infraestrutura de TI.\n"
            "Analise os seguintes itens não suportados e forneça:\n"
            "1. Diagnóstico da causa raiz de cada erro\n"
            "2. Impacto operacional considerando as triggers afetadas\n"
            "3. Passos detalhados para resolução (priorizado por severidade)\n"
            "4. Se o erro indica problema no agente, na rede, na configuração do item ou no host\n"
            "5. Estimativa de urgência: CRÍTICO / ALTO / MÉDIO / BAIXO\n\n"
            f"Dados dos itens não suportados:\n{json.dumps(items_data, ensure_ascii=False, indent=2)}\n\n"
            f"Triggers impactadas:\n{json.dumps(triggers_data, ensure_ascii=False, indent=2)}\n\n"
            "Responda em português brasileiro, de forma estruturada e técnica."
        )

        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.2, "maxOutputTokens": 8192},
        }
        url = f"{GEMINI_ENDPOINT}?key={self.api_key}"

        print(f"  → Chamando Gemini API (chamada {self.calls_made + 1}/{MAX_GEMINI_CALLS})...")
        try:
            data = http_post(url, payload)
            text = (
                data.get("candidates", [{}])[0]
                .get("content", {})
                .get("parts", [{}])[0]
                .get("text", "(resposta vazia)")
            )
            self.calls_made += 1
            self.cache[key] = text
            save_cache(self.cache)
            print(f"  ✓ Análise recebida ({len(text)} chars).")
            return text
        except Exception as exc:
            print(f"  ✗ Falha na Gemini API: {exc}")
            return f"(erro ao consultar Gemini: {exc})"


# ---------------------------------------------------------------------------
# Data processing helpers
# ---------------------------------------------------------------------------

PRIORITY_LABELS = {
    "0": "Não classificado",
    "1": "Informação",
    "2": "Aviso",
    "3": "Média",
    "4": "Alta",
    "5": "Desastre",
}

VALUE_LABELS = {"0": "OK ✅", "1": "PROBLEM 🔴"}


def fmt_ts(ts_str: str) -> str:
    try:
        ts = int(ts_str)
        if ts == 0:
            return "nunca"
        return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    except (ValueError, TypeError):
        return ts_str or "desconhecido"


def is_old(lastclock_str: str) -> bool:
    try:
        ts = int(lastclock_str)
        if ts == 0:
            return True
        age = time.time() - ts
        return age > UNSUPPORTED_OLD_DAYS * 86400
    except (ValueError, TypeError):
        return False


def group_by_error(items: list) -> dict:
    groups: dict = {}
    for item in items:
        err = (item.get("error") or "Erro desconhecido").strip()
        groups.setdefault(err, []).append(item)
    return groups


def build_trigger_index(triggers: list) -> dict:
    """Map itemid -> list of triggers."""
    index: dict = {}
    for trig in triggers:
        for it in trig.get("items", []):
            iid = it.get("itemid")
            if iid:
                index.setdefault(iid, []).append(trig)
    return index


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def render_trigger_table(triggers: list) -> str:
    if not triggers:
        return "_Nenhuma trigger associada._\n"
    lines = ["| Trigger | Severidade | Estado |", "|---------|-----------|--------|"]
    for t in triggers:
        desc = t.get("description", "—")
        prio = PRIORITY_LABELS.get(str(t.get("priority", "0")), "?")
        val = VALUE_LABELS.get(str(t.get("value", "0")), "?")
        lines.append(f"| {desc} | {prio} | {val} |")
    return "\n".join(lines) + "\n"


def render_item_block(item: dict, triggers: list, analysis: str = "") -> str:
    hosts = item.get("hosts", [{}])
    hostname = hosts[0].get("name") or hosts[0].get("host") or "Desconhecido"
    old_flag = " ⚠️ ANTIGO (>7 dias)" if is_old(item.get("lastclock", "0")) else ""

    block = f"""
### HOST: {hostname}
**Item:** {item.get('name', '—')}
**Key:** `{item.get('key_', '—')}`
**Erro Zabbix:** `{item.get('error', '—')}`
**Última tentativa:** {fmt_ts(item.get('lastclock', '0'))}{old_flag}
**Triggers cegas:**

{render_trigger_table(triggers)}
**🤖 Diagnóstico Gemini:**
{analysis if analysis else '_Análise de IA não disponível._'}

---"""
    return block


def generate_report(
    items: list,
    triggers: list,
    trigger_index: dict,
    group_analyses: dict,
    error_groups: dict,
) -> str:
    now = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    # Partition items
    items_online = [i for i in items if (i.get("hosts") or [{}])[0].get("status") == "0"]
    items_offline = [i for i in items if (i.get("hosts") or [{}])[0].get("status") != "0"]

    active_problem_ids = {
        iid
        for iid, trigs in trigger_index.items()
        if any(str(t.get("value")) == "1" for t in trigs)
    }

    critical_items = [i for i in items_online if i["itemid"] in active_problem_ids]
    alert_items = [i for i in items_online if i["itemid"] not in active_problem_ids]

    unique_hosts = len({(i.get("hosts") or [{}])[0].get("hostid") for i in items})
    blind_triggers = sum(len(v) for v in trigger_index.values())
    active_problems = sum(
        1 for trigs in trigger_index.values()
        for t in trigs if str(t.get("value")) == "1"
    )

    lines = [
        f"# RELATÓRIO DE ITENS NÃO SUPORTADOS — ZABBIX",
        f"**Gerado em:** {now}  ",
        f"**Analisado por:** Google Gemini 2.5 Flash  ",
        f"**Total de itens não suportados:** {len(items)}  ",
        f"**Hosts afetados:** {unique_hosts}  ",
        f"**Triggers potencialmente cegas:** {blind_triggers}  ",
        f"**Triggers em PROBLEM ativo:** {active_problems}  ",
        "",
        "---",
        "",
        "## 🔴 CRÍTICOS (Triggers ativas em PROBLEM)",
        "",
    ]

    def get_analysis(item: dict) -> str:
        err = (item.get("error") or "").strip()
        return group_analyses.get(err, "")

    if critical_items:
        for item in critical_items:
            trigs = trigger_index.get(item["itemid"], [])
            lines.append(render_item_block(item, trigs, get_analysis(item)))
    else:
        lines.append("_Nenhum item crítico com trigger em PROBLEM ativo._\n")

    lines += ["", "## 🟡 ALERTAS (Itens sem suporte mas triggers OK)", ""]
    if alert_items:
        for item in alert_items:
            trigs = trigger_index.get(item["itemid"], [])
            lines.append(render_item_block(item, trigs, get_analysis(item)))
    else:
        lines.append("_Nenhum item de alerta._\n")

    if items_offline:
        lines += ["", "## 🔌 HOSTS OFFLINE", ""]
        for item in items_offline:
            trigs = trigger_index.get(item["itemid"], [])
            lines.append(render_item_block(item, trigs, get_analysis(item)))

    # Summary table
    lines += ["", "## 📊 RESUMO POR TIPO DE ERRO", ""]
    lines.append("| Erro | Qtd Itens | Hosts | Urgência Estimada |")
    lines.append("|------|-----------|-------|-------------------|")
    for err, grp in error_groups.items():
        host_count = len({(i.get("hosts") or [{}])[0].get("hostid") for i in grp})
        analysis = group_analyses.get(err, "")
        urgency = "DESCONHECIDA"
        for level in ["CRÍTICO", "ALTO", "MÉDIO", "BAIXO"]:
            if level in analysis.upper():
                urgency = level
                break
        lines.append(f"| {err[:80]} | {len(grp)} | {host_count} | {urgency} |")

    # Action plan
    lines += ["", "## ✅ PLANO DE AÇÃO (gerado pela IA)", ""]
    if group_analyses:
        for err, analysis in group_analyses.items():
            lines.append(f"### Erro: `{err[:80]}`\n")
            lines.append(analysis)
            lines.append("")
    else:
        lines.append("_Análise de IA não disponível._")

    lines += ["", "---", f"_Relatório gerado automaticamente por zabbix_diagnostic.py_"]
    return "\n".join(lines)


def executive_summary(items: list, triggers: list, trigger_index: dict):
    unique_hosts = len({(i.get("hosts") or [{}])[0].get("hostid") for i in items})
    active_problems = sum(
        1 for trigs in trigger_index.values()
        for t in trigs if str(t.get("value")) == "1"
    )
    old_count = sum(1 for i in items if is_old(i.get("lastclock", "0")))

    print()
    print("=" * 60)
    print("RESUMO EXECUTIVO")
    print("=" * 60)
    print(f"1. Total de itens não suportados: {len(items)} em {unique_hosts} host(s).")
    print(f"2. Triggers potencialmente cegas: {len(triggers)} (monitoramento comprometido).")
    print(f"3. Triggers atualmente em PROBLEM com item sem suporte: {active_problems}.")
    print(f"4. Itens sem suporte há mais de {UNSUPPORTED_OLD_DAYS} dias (ANTIGOS): {old_count}.")
    print(f"5. Relatório detalhado salvo em: {REPORT_FILE.resolve()}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print()
    print("=" * 60)
    print("  ZABBIX DIAGNOSTIC AGENT — Powered by Google Gemini")
    print("=" * 60)
    print()

    env = load_env()

    # Step 1 — Auth
    zabbix = ZabbixClient(env["ZABBIX_URL"], env["ZABBIX_USER"], env["ZABBIX_PASSWORD"])
    zabbix.login()

    # Step 2 — Unsupported items
    items = zabbix.get_unsupported_items()
    if not items:
        print("\n✓ Nenhum item não suportado encontrado. Ambiente saudável!")
        return

    item_ids = [i["itemid"] for i in items]

    # Step 3 — Triggers
    triggers = zabbix.get_triggers_for_items(item_ids)
    trigger_index = build_trigger_index(triggers)

    # Step 4 — Gemini analysis per error group
    print()
    print("→ Iniciando análise com Gemini AI...")
    gemini = GeminiClient(env["GEMINI_API_KEY"])
    error_groups = group_by_error(items)
    group_analyses: dict = {}

    for err, grp in error_groups.items():
        print(f"\n  Grupo de erro: '{err[:60]}...' ({len(grp)} item(s))")
        # Batch into chunks of MAX_ITEMS_PER_GEMINI_CALL
        for batch_start in range(0, len(grp), MAX_ITEMS_PER_GEMINI_CALL):
            batch = grp[batch_start: batch_start + MAX_ITEMS_PER_GEMINI_CALL]
            # Gather related triggers for this batch
            batch_item_ids = {i["itemid"] for i in batch}
            batch_triggers = [
                t for iid, trigs in trigger_index.items()
                if iid in batch_item_ids
                for t in trigs
            ]
            analysis = gemini.analyze(batch, batch_triggers)
            # Merge analysis (last batch wins if multiple chunks)
            group_analyses[err] = analysis
            if gemini.calls_made >= MAX_GEMINI_CALLS:
                print("  ⚠ Limite de chamadas Gemini atingido.")
                break
        if gemini.calls_made >= MAX_GEMINI_CALLS:
            break

    # Step 5 — Report
    print()
    print("→ Gerando relatório Markdown...")
    report = generate_report(items, triggers, trigger_index, group_analyses, error_groups)
    REPORT_FILE.write_text(report, encoding="utf-8")
    print(f"  ✓ Relatório salvo: {REPORT_FILE.resolve()}")

    # Executive summary
    executive_summary(items, triggers, trigger_index)


if __name__ == "__main__":
    main()
