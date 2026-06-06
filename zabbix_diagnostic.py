#!/usr/bin/env python3
"""
Zabbix Unsupported Items Diagnostic Agent with Google Gemini AI Analysis.

Requirements: requests (pip install requests)

Authentication — set ONE of:
    ZABBIX_TOKEN     - Zabbix API token (preferred, Settings > API tokens)
    ZABBIX_USER + ZABBIX_PASSWORD - username/password fallback

Other required:
    ZABBIX_URL       - Zabbix base URL (e.g. https://zabbix.empresa.com.br)
    GEMINI_API_KEY   - Google Gemini API key (free at aistudio.google.com)

Scale strategy for large environments (60k+ items):
    - Items are fetched in pages of PAGE_SIZE via cursor pagination
    - Grouped by normalized error message (~10-50 distinct groups in practice)
    - Only SAMPLE_PER_GROUP representative items are sent to Gemini per group
    - Gemini diagnosis is applied to ALL items in that error group
    - This keeps Gemini calls bounded regardless of total item count
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

PAGE_SIZE = 1000          # items fetched per Zabbix API call
SAMPLE_PER_GROUP = 5      # representative items sent to Gemini per error group
MAX_GEMINI_CALLS = 20     # max Gemini calls per run (free tier: 1500/day)
UNSUPPORTED_OLD_DAYS = 7


# ---------------------------------------------------------------------------
# Environment validation
# ---------------------------------------------------------------------------

def load_env() -> dict:
    cfg = {
        "ZABBIX_URL": os.environ.get("ZABBIX_URL", "").rstrip("/"),
        "ZABBIX_TOKEN": os.environ.get("ZABBIX_TOKEN", ""),
        "ZABBIX_USER": os.environ.get("ZABBIX_USER", ""),
        "ZABBIX_PASSWORD": os.environ.get("ZABBIX_PASSWORD", ""),
        "GEMINI_API_KEY": os.environ.get("GEMINI_API_KEY", ""),
    }

    errors = []
    if not cfg["ZABBIX_URL"]:
        errors.append("ZABBIX_URL")
    if not cfg["GEMINI_API_KEY"]:
        errors.append("GEMINI_API_KEY")
    if not cfg["ZABBIX_TOKEN"] and not (cfg["ZABBIX_USER"] and cfg["ZABBIX_PASSWORD"]):
        errors.append("ZABBIX_TOKEN (ou ZABBIX_USER + ZABBIX_PASSWORD)")

    if errors:
        print("=" * 60)
        print("ERRO: Variáveis de ambiente obrigatórias não definidas:")
        for var in errors:
            print(f"  → {var}")
        print()
        print("Opção 1 — autenticação por token (recomendado):")
        print('  export ZABBIX_URL="https://zabbix.sua-empresa.com.br"')
        print('  export ZABBIX_TOKEN="seu_api_token"')
        print('  export GEMINI_API_KEY="sua_chave_gemini"')
        print()
        print("Opção 2 — autenticação por usuário/senha:")
        print('  export ZABBIX_URL="https://zabbix.sua-empresa.com.br"')
        print('  export ZABBIX_USER="Admin"')
        print('  export ZABBIX_PASSWORD="sua_senha"')
        print('  export GEMINI_API_KEY="sua_chave_gemini"')
        print()
        print("Token Zabbix: Administration > API tokens > Create API token")
        print("Chave Gemini gratuita em: https://aistudio.google.com/")
        print("=" * 60)
        sys.exit(1)

    return cfg


# ---------------------------------------------------------------------------
# HTTP helpers with retry + exponential backoff
# ---------------------------------------------------------------------------

def http_post(url: str, payload: dict, headers: dict = None, max_retries: int = 3) -> dict:
    headers = headers or {"Content-Type": "application/json"}
    for attempt in range(max_retries):
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=60)
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
# Zabbix API client
# ---------------------------------------------------------------------------

class ZabbixClient:
    def __init__(self, url: str, token: str = "", user: str = "", password: str = ""):
        self.api_url = f"{url}/api_jsonrpc.php"
        self._token = token          # static API token (no expiry, no login call)
        self._user = user
        self._password = password
        self._session_token: str = ""  # obtained via user.login if no static token
        self._req_id = 1

    # ------------------------------------------------------------------
    # Auth
    # ------------------------------------------------------------------

    def _auth_header(self) -> dict:
        """Return Authorization header for token-based auth (Zabbix ≥ 5.4)."""
        token = self._token or self._session_token
        return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    def login(self):
        if self._token:
            print("→ Autenticação via API token (Bearer)...")
            # Validate token with a lightweight call
            try:
                self._call_raw("apiinfo.version", {})
                print("  ✓ Token válido.")
            except Exception as exc:
                print(f"  ✗ Token inválido ou URL incorreta: {exc}")
                sys.exit(1)
        else:
            print("→ Autenticando via usuário/senha...")
            result = self._call_raw("user.login", {
                "username": self._user,
                "password": self._password,
            }, auth=False)
            self._session_token = result
            print(f"  ✓ Autenticado (token de sessão: {result[:8]}...)")

    # ------------------------------------------------------------------
    # Internal call
    # ------------------------------------------------------------------

    def _call_raw(self, method: str, params: dict, auth: bool = True) -> any:
        payload = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
            "id": self._req_id,
        }
        self._req_id += 1

        if auth and self._session_token and not self._token:
            # Legacy session token passed in payload (Zabbix < 5.4 compatibility)
            payload["auth"] = self._session_token

        headers = self._auth_header() if auth else {"Content-Type": "application/json"}
        data = http_post(self.api_url, payload, headers=headers)

        if "error" in data:
            raise RuntimeError(f"Zabbix API error [{method}]: {data['error']}")
        return data.get("result")

    def _call(self, method: str, params: dict) -> any:
        return self._call_raw(method, params, auth=True)

    # ------------------------------------------------------------------
    # Data fetching
    # ------------------------------------------------------------------

    def get_unsupported_items(self) -> list:
        """Fetch ALL unsupported items using cursor/offset pagination."""
        print("→ Buscando itens não suportados (state=1) com paginação...")
        all_items = []
        offset = 0

        while True:
            batch = self._call("item.get", {
                "output": ["itemid", "name", "key_", "error", "lastclock",
                           "hostid", "status", "state"],
                "filter": {"state": 1},
                "selectHosts": ["hostid", "host", "name", "status"],
                "limit": PAGE_SIZE,
                "limitstart": offset,
                "sortfield": "itemid",
                "sortorder": "ASC",
            })
            if not batch:
                break
            all_items.extend(batch)
            print(f"  … {len(all_items)} itens carregados", end="\r")
            if len(batch) < PAGE_SIZE:
                break
            offset += PAGE_SIZE

        print(f"  ✓ {len(all_items)} itens não suportados encontrados.{' ' * 20}")
        return all_items

    def get_triggers_for_items(self, item_ids: list) -> list:
        """Fetch triggers in batches to avoid oversized requests."""
        if not item_ids:
            return []

        BATCH = 500
        all_triggers = []
        print(f"→ Buscando triggers ({len(item_ids)} itens, lotes de {BATCH})...")

        for i in range(0, len(item_ids), BATCH):
            chunk = item_ids[i: i + BATCH]
            result = self._call("trigger.get", {
                "output": ["triggerid", "description", "priority", "value", "lastchange"],
                "itemids": chunk,
                "selectItems": ["itemid", "name"],
                "filter": {"status": 0},
                "expandDescription": True,
            })
            all_triggers.extend(result)

        # Deduplicate by triggerid
        seen = set()
        unique = []
        for t in all_triggers:
            tid = t.get("triggerid")
            if tid not in seen:
                seen.add(tid)
                unique.append(t)

        print(f"  ✓ {len(unique)} triggers únicas encontradas.")
        return unique


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


def cache_key(error_signature: str) -> str:
    return hashlib.sha256(error_signature.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Gemini API
# ---------------------------------------------------------------------------

class GeminiClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.calls_made = 0
        self.cache = load_cache()

    def analyze_group(self, error_msg: str, sample_items: list,
                      sample_triggers: list, total_count: int, host_count: int) -> str:
        """
        Analyze one error group using only a sample.
        The diagnosis applies to all items sharing the same error.
        """
        key = cache_key(error_msg)
        if key in self.cache:
            print(f"  ↩ Cache hit (key={key})")
            return self.cache[key]

        if self.calls_made >= MAX_GEMINI_CALLS:
            return "(limite de chamadas Gemini atingido — análise não disponível)"

        prompt = (
            "Você é um especialista em monitoramento Zabbix e infraestrutura de TI.\n\n"
            f"CONTEXTO DO GRUPO DE ERRO:\n"
            f"- Mensagem de erro: \"{error_msg}\"\n"
            f"- Total de itens afetados com ESTE erro: {total_count}\n"
            f"- Total de hosts afetados: {host_count}\n"
            f"- Amostra de {len(sample_items)} itens representativos (o diagnóstico deve "
            f"ser aplicável a todos os {total_count} itens com este erro):\n\n"
            f"{json.dumps(sample_items, ensure_ascii=False, indent=2)}\n\n"
            f"Triggers associadas (amostra):\n"
            f"{json.dumps(sample_triggers, ensure_ascii=False, indent=2)}\n\n"
            "Forneça:\n"
            "1. **Diagnóstico da causa raiz** — o que este erro significa tecnicamente\n"
            "2. **Categoria do problema** — agente Zabbix / rede / configuração do item / "
            "host inacessível / permissão / outro\n"
            "3. **Impacto operacional** — o que deixa de ser monitorado\n"
            "4. **Passos de resolução** — comandos e ações concretas, em ordem de prioridade\n"
            "5. **Urgência geral do grupo**: CRÍTICO / ALTO / MÉDIO / BAIXO\n\n"
            "Responda em português brasileiro, de forma estruturada e técnica. "
            "Seja direto — o diagnóstico será aplicado a todos os itens do grupo."
        )

        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.2, "maxOutputTokens": 4096},
        }
        url = f"{GEMINI_ENDPOINT}?key={self.api_key}"

        print(f"  → Gemini API chamada {self.calls_made + 1}/{MAX_GEMINI_CALLS} "
              f"(grupo: '{error_msg[:50]}...')")
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
        return (time.time() - ts) > UNSUPPORTED_OLD_DAYS * 86400
    except (ValueError, TypeError):
        return False


def normalize_error(error: str) -> str:
    """
    Normalize error message to group semantically identical errors.
    E.g. 'Cannot connect to 192.168.1.10' and 'Cannot connect to 10.0.0.5'
    both map to 'Cannot connect to <host>'.
    """
    import re
    msg = (error or "Erro desconhecido").strip()
    # Replace IPs
    msg = re.sub(r"\b\d{1,3}(\.\d{1,3}){3}\b", "<IP>", msg)
    # Replace port numbers after colon
    msg = re.sub(r":\d{2,5}\b", ":<PORT>", msg)
    # Replace OIDs
    msg = re.sub(r"\.\d+(\.\d+){3,}", ".<OID>", msg)
    # Replace quoted strings (item key parameters)
    msg = re.sub(r'"[^"]{20,}"', '"<VALUE>"', msg)
    return msg


def group_items_by_error(items: list) -> dict[str, dict]:
    """
    Returns: { normalized_error: { "items": [...], "raw_error": str } }
    """
    groups: dict = {}
    for item in items:
        raw = (item.get("error") or "Erro desconhecido").strip()
        key = normalize_error(raw)
        if key not in groups:
            groups[key] = {"raw_error": raw, "items": []}
        groups[key]["items"].append(item)
    return groups


def build_trigger_index(triggers: list) -> dict:
    index: dict = {}
    for trig in triggers:
        for it in trig.get("items", []):
            iid = it.get("itemid")
            if iid:
                index.setdefault(iid, []).append(trig)
    return index


def extract_urgency(analysis: str) -> str:
    for level in ["CRÍTICO", "ALTO", "MÉDIO", "BAIXO"]:
        if level in analysis.upper():
            return level
    return "DESCONHECIDA"


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def render_trigger_table(triggers: list) -> str:
    if not triggers:
        return "_Nenhuma trigger associada._\n"
    lines = ["| Trigger | Severidade | Estado |", "|---------|-----------|--------|"]
    for t in triggers[:10]:  # cap at 10 per item for readability
        desc = t.get("description", "—")
        prio = PRIORITY_LABELS.get(str(t.get("priority", "0")), "?")
        val = VALUE_LABELS.get(str(t.get("value", "0")), "?")
        lines.append(f"| {desc} | {prio} | {val} |")
    return "\n".join(lines) + "\n"


def generate_report(
    items: list,
    triggers: list,
    trigger_index: dict,
    error_groups: dict,
    group_analyses: dict,
) -> str:
    now = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

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

    # Map itemid → normalized error group key
    item_to_group: dict = {}
    for grp_key, grp in error_groups.items():
        for item in grp["items"]:
            item_to_group[item["itemid"]] = grp_key

    def item_analysis(item: dict) -> str:
        grp_key = item_to_group.get(item["itemid"], "")
        return group_analyses.get(grp_key, "")

    def item_block(item: dict) -> str:
        hosts = item.get("hosts", [{}])
        hostname = hosts[0].get("name") or hosts[0].get("host") or "Desconhecido"
        old_flag = " ⚠️ ANTIGO (>7 dias)" if is_old(item.get("lastclock", "0")) else ""
        trigs = trigger_index.get(item["itemid"], [])
        analysis = item_analysis(item)
        return (
            f"\n### HOST: {hostname}\n"
            f"**Item:** {item.get('name', '—')}  \n"
            f"**Key:** `{item.get('key_', '—')}`  \n"
            f"**Erro Zabbix:** `{item.get('error', '—')}`  \n"
            f"**Última tentativa:** {fmt_ts(item.get('lastclock', '0'))}{old_flag}  \n"
            f"**Triggers cegas:**\n\n{render_trigger_table(trigs)}\n"
            f"**🤖 Diagnóstico Gemini:**\n"
            f"{analysis if analysis else '_Análise de IA não disponível para este grupo._'}\n\n---"
        )

    lines = [
        "# RELATÓRIO DE ITENS NÃO SUPORTADOS — ZABBIX",
        f"**Gerado em:** {now}  ",
        "**Analisado por:** Google Gemini 2.5 Flash  ",
        f"**Total de itens não suportados:** {len(items)}  ",
        f"**Hosts afetados:** {unique_hosts}  ",
        f"**Grupos de erro distintos:** {len(error_groups)}  ",
        f"**Triggers potencialmente cegas:** {blind_triggers}  ",
        f"**Triggers em PROBLEM ativo:** {active_problems}  ",
        "",
        "---",
        "",
        "## 🔴 CRÍTICOS (Triggers ativas em PROBLEM)",
        "",
    ]

    if critical_items:
        for item in critical_items:
            lines.append(item_block(item))
    else:
        lines.append("_Nenhum item crítico com trigger em PROBLEM ativo._\n")

    lines += ["", "## 🟡 ALERTAS (Itens sem suporte, triggers OK)", ""]
    if alert_items:
        # For very large sets, cap individual item blocks and show a summary table
        BLOCK_LIMIT = 200
        for item in alert_items[:BLOCK_LIMIT]:
            lines.append(item_block(item))
        if len(alert_items) > BLOCK_LIMIT:
            lines.append(
                f"\n> **{len(alert_items) - BLOCK_LIMIT} itens adicionais** não exibidos "
                f"individualmente. Consulte o resumo por tipo de erro abaixo.\n"
            )
    else:
        lines.append("_Nenhum item de alerta._\n")

    if items_offline:
        lines += ["", "## 🔌 HOSTS OFFLINE", ""]
        for item in items_offline[:50]:
            lines.append(item_block(item))

    # Summary by error group
    lines += ["", "## 📊 RESUMO POR GRUPO DE ERRO", ""]
    lines.append("| Erro (normalizado) | Itens | Hosts | Urgência | Gemini |")
    lines.append("|--------------------|-------|-------|----------|--------|")
    for grp_key, grp in sorted(error_groups.items(),
                                key=lambda x: len(x[1]["items"]), reverse=True):
        grp_items = grp["items"]
        h_count = len({(i.get("hosts") or [{}])[0].get("hostid") for i in grp_items})
        analysis = group_analyses.get(grp_key, "")
        urgency = extract_urgency(analysis) if analysis else "—"
        cached = "✓ cache" if cache_key(grp_key) in load_cache() else "✓" if analysis else "✗"
        lines.append(
            f"| {grp_key[:70]} | {len(grp_items)} | {h_count} | {urgency} | {cached} |"
        )

    # Action plan
    lines += ["", "## ✅ PLANO DE AÇÃO CONSOLIDADO (gerado pela IA)", ""]
    if group_analyses:
        # Sort by urgency: CRÍTICO first
        order = {"CRÍTICO": 0, "ALTO": 1, "MÉDIO": 2, "BAIXO": 3, "DESCONHECIDA": 4}
        sorted_groups = sorted(
            group_analyses.items(),
            key=lambda x: order.get(extract_urgency(x[1]), 4),
        )
        for grp_key, analysis in sorted_groups:
            grp = error_groups.get(grp_key, {})
            count = len(grp.get("items", []))
            urgency = extract_urgency(analysis)
            lines.append(f"### [{urgency}] `{grp_key[:80]}` — {count} item(s)\n")
            lines.append(analysis)
            lines.append("")
    else:
        lines.append("_Análise de IA não disponível._")

    lines += [
        "",
        "---",
        "_Relatório gerado automaticamente por zabbix_diagnostic.py_",
        f"_{len(error_groups)} grupos de erro analisados | "
        f"{len(items)} itens totais | {unique_hosts} hosts_",
    ]
    return "\n".join(lines)


def executive_summary(items: list, triggers: list, trigger_index: dict,
                      error_groups: dict, group_analyses: dict):
    unique_hosts = len({(i.get("hosts") or [{}])[0].get("hostid") for i in items})
    active_problems = sum(
        1 for trigs in trigger_index.values()
        for t in trigs if str(t.get("value")) == "1"
    )
    old_count = sum(1 for i in items if is_old(i.get("lastclock", "0")))
    critical_groups = [
        k for k, v in group_analyses.items() if extract_urgency(v) == "CRÍTICO"
    ]

    print()
    print("=" * 60)
    print("RESUMO EXECUTIVO")
    print("=" * 60)
    print(f"1. {len(items)} itens não suportados em {unique_hosts} host(s) "
          f"({len(error_groups)} grupos de erro distintos).")
    print(f"2. {len(triggers)} triggers potencialmente cegas "
          f"(monitoramento comprometido).")
    print(f"3. {active_problems} trigger(s) em PROBLEM ativo com item sem suporte.")
    print(f"4. {old_count} item(s) sem suporte há mais de {UNSUPPORTED_OLD_DAYS} dias.")
    if critical_groups:
        print(f"5. ⚠ {len(critical_groups)} grupo(s) de erro classificados como CRÍTICO "
              f"pela IA — ação imediata recomendada.")
    else:
        print(f"5. Relatório completo salvo em: {REPORT_FILE.resolve()}")
    print("=" * 60)
    print(f"\nRelatório: {REPORT_FILE.resolve()}")


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
    zabbix = ZabbixClient(
        url=env["ZABBIX_URL"],
        token=env["ZABBIX_TOKEN"],
        user=env["ZABBIX_USER"],
        password=env["ZABBIX_PASSWORD"],
    )
    zabbix.login()

    # Step 2 — Fetch all unsupported items (paginated)
    items = zabbix.get_unsupported_items()
    if not items:
        print("\n✓ Nenhum item não suportado encontrado. Ambiente saudável!")
        return

    # Step 3 — Group by normalized error (BEFORE fetching triggers)
    print(f"\n→ Agrupando {len(items)} itens por tipo de erro...")
    error_groups = group_items_by_error(items)
    print(f"  ✓ {len(error_groups)} grupos de erro distintos identificados.")
    for grp_key, grp in sorted(error_groups.items(),
                                key=lambda x: -len(x[1]["items"]))[:10]:
        print(f"    [{len(grp['items']):>5} itens] {grp_key[:70]}")
    if len(error_groups) > 10:
        print(f"    ... e mais {len(error_groups) - 10} grupo(s).")

    # Step 4 — Triggers (only for items that actually have triggers — use sample)
    # To avoid a 60k-itemid request, fetch triggers per group using samples
    sample_item_ids = []
    for grp in error_groups.values():
        sample_item_ids.extend(i["itemid"] for i in grp["items"][:SAMPLE_PER_GROUP])
    sample_item_ids = list(set(sample_item_ids))

    triggers = zabbix.get_triggers_for_items(sample_item_ids)
    trigger_index = build_trigger_index(triggers)

    # Step 5 — Gemini analysis: one call per error group (using samples)
    print(f"\n→ Analisando {len(error_groups)} grupo(s) com Gemini AI "
          f"(máx {SAMPLE_PER_GROUP} itens de amostra por grupo)...")
    gemini = GeminiClient(env["GEMINI_API_KEY"])
    group_analyses: dict = {}

    for grp_key, grp in sorted(error_groups.items(),
                                key=lambda x: -len(x[1]["items"])):
        grp_items = grp["items"]
        sample = grp_items[:SAMPLE_PER_GROUP]
        host_count = len({(i.get("hosts") or [{}])[0].get("hostid") for i in grp_items})

        # Triggers for the sample items
        sample_ids = {i["itemid"] for i in sample}
        sample_triggers = list({
            t["triggerid"]: t
            for iid, trigs in trigger_index.items()
            if iid in sample_ids
            for t in trigs
        }.values())[:10]

        analysis = gemini.analyze_group(
            error_msg=grp_key,
            sample_items=sample,
            sample_triggers=sample_triggers,
            total_count=len(grp_items),
            host_count=host_count,
        )
        group_analyses[grp_key] = analysis

        if gemini.calls_made >= MAX_GEMINI_CALLS:
            remaining = len(error_groups) - len(group_analyses)
            if remaining > 0:
                print(f"  ⚠ Limite de {MAX_GEMINI_CALLS} chamadas atingido. "
                      f"{remaining} grupo(s) sem análise de IA.")
            break

    # Step 6 — Report
    print("\n→ Gerando relatório Markdown...")
    report = generate_report(items, triggers, trigger_index, error_groups, group_analyses)
    REPORT_FILE.write_text(report, encoding="utf-8")
    print(f"  ✓ Relatório salvo: {REPORT_FILE.resolve()}")

    executive_summary(items, triggers, trigger_index, error_groups, group_analyses)


if __name__ == "__main__":
    main()
