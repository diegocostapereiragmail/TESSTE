#!/usr/bin/env python3
"""
Zabbix Unsupported Items Diagnostic Agent with Google Gemini AI Analysis.

Requirements: requests (pip install requests)

Authentication — set ONE of:
    ZABBIX_TOKEN                    - Zabbix API token (preferred)
    ZABBIX_USER + ZABBIX_PASSWORD   - username/password fallback

Required:
    ZABBIX_URL      - Zabbix base URL (https://zabbix.empresa.com.br)
    GEMINI_API_KEY  - Google Gemini API key (free at aistudio.google.com)

Email alerts (optional — script still runs without these):
    SMTP_HOST       - SMTP server hostname (e.g. smtp.gmail.com)
    SMTP_PORT       - SMTP port (default: 587)
    SMTP_USER       - SMTP username / sender address
    SMTP_PASSWORD   - SMTP password or app password
    ALERT_EMAIL     - recipient address (comma-separated for multiple)

Usage:
    python3 zabbix_diagnostic.py                  # run once
    python3 zabbix_diagnostic.py --watch 10       # run every 10 minutes
    python3 zabbix_diagnostic.py --watch 10 --no-gemini  # fast mode, no AI

Intermittent item detection:
    State is persisted in .zabbix_state.json between runs.
    Email is sent ONLY when items change state (new unsupported or recovered),
    which is ideal for catching flapping / intermittent items.
"""

import os
import sys
import json
import time
import hashlib
import smtplib
import argparse
import re
from datetime import datetime, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
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
CACHE_FILE   = Path(".zabbix_gemini_cache.json")
STATE_FILE   = Path(".zabbix_state.json")
REPORT_MD    = Path("zabbix_unsupported_report.md")
REPORT_HTML  = Path("zabbix_unsupported_report.html")

PAGE_SIZE           = 1000
SAMPLE_PER_GROUP    = 5
MAX_GEMINI_CALLS    = 20
UNSUPPORTED_OLD_DAYS = 7


# ---------------------------------------------------------------------------
# Environment validation
# ---------------------------------------------------------------------------

def load_env() -> dict:
    cfg = {
        "ZABBIX_URL":      os.environ.get("ZABBIX_URL", "").rstrip("/"),
        "ZABBIX_TOKEN":    os.environ.get("ZABBIX_TOKEN", ""),
        "ZABBIX_USER":     os.environ.get("ZABBIX_USER", ""),
        "ZABBIX_PASSWORD": os.environ.get("ZABBIX_PASSWORD", ""),
        "GEMINI_API_KEY":  os.environ.get("GEMINI_API_KEY", ""),
        # Email (optional)
        "SMTP_HOST":       os.environ.get("SMTP_HOST", ""),
        "SMTP_PORT":       int(os.environ.get("SMTP_PORT", "587")),
        "SMTP_USER":       os.environ.get("SMTP_USER", ""),
        "SMTP_PASSWORD":   os.environ.get("SMTP_PASSWORD", ""),
        "ALERT_EMAIL":     os.environ.get("ALERT_EMAIL", ""),
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
        print("Configuração rápida:")
        print('  export ZABBIX_URL="https://zabbix.empresa.com.br"')
        print('  export ZABBIX_TOKEN="seu_api_token"')
        print('  export GEMINI_API_KEY="sua_chave_gemini"')
        print()
        print("E-mail de alerta (opcional):")
        print('  export SMTP_HOST="smtp.gmail.com"')
        print('  export SMTP_PORT="587"')
        print('  export SMTP_USER="seu@email.com"')
        print('  export SMTP_PASSWORD="senha_ou_app_password"')
        print('  export ALERT_EMAIL="destino@email.com"')
        print()
        print("Token Zabbix: Administration > API tokens > Create API token")
        print("Chave Gemini gratuita: https://aistudio.google.com/")
        print("=" * 60)
        sys.exit(1)

    return cfg


# ---------------------------------------------------------------------------
# HTTP helpers
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
                print(f"  [retry {attempt + 1}/{max_retries}] {exc}. Aguardando {wait}s...")
                time.sleep(wait)
            else:
                raise


# ---------------------------------------------------------------------------
# Zabbix API client
# ---------------------------------------------------------------------------

class ZabbixClient:
    def __init__(self, url, token="", user="", password=""):
        self.api_url = f"{url}/api_jsonrpc.php"
        self._token = token
        self._user = user
        self._password = password
        self._session_token = ""
        self._req_id = 1

    def _auth_header(self):
        token = self._token or self._session_token
        return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    def login(self):
        if self._token:
            print("→ Autenticação via API token (Bearer)...")
            try:
                self._call_raw("apiinfo.version", {})
                print("  ✓ Token válido.")
            except Exception as exc:
                print(f"  ✗ Token inválido: {exc}")
                sys.exit(1)
        else:
            print("→ Autenticando via usuário/senha...")
            result = self._call_raw("user.login", {
                "username": self._user,
                "password": self._password,
            }, auth=False)
            self._session_token = result
            print(f"  ✓ Autenticado (sessão: {result[:8]}...)")

    def _call_raw(self, method, params, auth=True):
        payload = {"jsonrpc": "2.0", "method": method, "params": params, "id": self._req_id}
        self._req_id += 1
        if auth and self._session_token and not self._token:
            payload["auth"] = self._session_token
        headers = self._auth_header() if auth else {"Content-Type": "application/json"}
        data = http_post(self.api_url, payload, headers=headers)
        if "error" in data:
            raise RuntimeError(f"Zabbix API [{method}]: {data['error']}")
        return data.get("result")

    def _call(self, method, params):
        return self._call_raw(method, params, auth=True)

    def get_unsupported_items(self):
        print("→ Buscando itens não suportados (paginação)...")
        all_items, offset = [], 0
        while True:
            batch = self._call("item.get", {
                "output": ["itemid", "name", "key_", "error", "lastclock", "hostid", "status", "state"],
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
            print(f"  … {len(all_items)} itens", end="\r")
            if len(batch) < PAGE_SIZE:
                break
            offset += PAGE_SIZE
        print(f"  ✓ {len(all_items)} itens não suportados.{' ' * 20}")
        return all_items

    def get_triggers_for_items(self, item_ids):
        if not item_ids:
            return []
        BATCH = 500
        all_triggers = []
        print(f"→ Buscando triggers ({len(item_ids)} itens)...")
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
        seen, unique = set(), []
        for t in all_triggers:
            if t["triggerid"] not in seen:
                seen.add(t["triggerid"])
                unique.append(t)
        print(f"  ✓ {len(unique)} triggers únicas.")
        return unique


# ---------------------------------------------------------------------------
# State tracking (intermittent item detection)
# ---------------------------------------------------------------------------

def load_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return {"items": {}, "last_run": None}


def save_state(state: dict):
    STATE_FILE.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def detect_changes(current_items: list, state: dict) -> dict:
    """
    Compare current unsupported items against previous state.
    Returns new items, recovered items, and persistent items.
    """
    now_ts = int(time.time())
    now_ids = {i["itemid"]: i for i in current_items}
    prev_ids: dict = state.get("items", {})

    new_items = [
        now_ids[iid] for iid in now_ids
        if iid not in prev_ids
    ]
    recovered_items = [
        prev_ids[iid] for iid in prev_ids
        if iid not in now_ids
    ]
    persistent_items = [
        now_ids[iid] for iid in now_ids
        if iid in prev_ids
    ]

    # Update state
    new_state = {
        "items": {
            iid: {
                "itemid": iid,
                "name": item.get("name"),
                "key_": item.get("key_"),
                "error": item.get("error"),
                "host": (item.get("hosts") or [{}])[0].get("name", "?"),
                "first_seen": prev_ids.get(iid, {}).get("first_seen", now_ts),
                "last_seen": now_ts,
                "seen_count": prev_ids.get(iid, {}).get("seen_count", 0) + 1,
            }
            for iid, item in now_ids.items()
        },
        "last_run": now_ts,
    }

    return {
        "new": new_items,
        "recovered": recovered_items,
        "persistent": persistent_items,
        "new_state": new_state,
        "has_changes": bool(new_items or recovered_items),
    }


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


def cache_key(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Gemini API
# ---------------------------------------------------------------------------

class GeminiClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.calls_made = 0
        self.cache = load_cache()

    def analyze_group(self, error_msg, sample_items, sample_triggers, total_count, host_count):
        key = cache_key(error_msg)
        if key in self.cache:
            print(f"  ↩ Cache hit (key={key})")
            return self.cache[key]

        if self.calls_made >= MAX_GEMINI_CALLS:
            return "(limite de chamadas Gemini atingido — análise não disponível)"

        prompt = (
            "Você é um especialista em monitoramento Zabbix e infraestrutura de TI.\n\n"
            f"GRUPO DE ERRO:\n"
            f"- Mensagem: \"{error_msg}\"\n"
            f"- Total de itens afetados: {total_count} em {host_count} host(s)\n"
            f"- Amostra representativa ({len(sample_items)} itens):\n"
            f"{json.dumps(sample_items, ensure_ascii=False, indent=2)}\n\n"
            f"Triggers associadas (amostra):\n"
            f"{json.dumps(sample_triggers, ensure_ascii=False, indent=2)}\n\n"
            "Forneça:\n"
            "1. **Diagnóstico da causa raiz** — o que este erro significa tecnicamente\n"
            "2. **Categoria** — agente Zabbix / rede / configuração / host inacessível / permissão\n"
            "3. **Impacto operacional** — o que deixa de ser monitorado\n"
            "4. **Passos de resolução** — comandos concretos em ordem de prioridade\n"
            "5. **Urgência**: CRÍTICO / ALTO / MÉDIO / BAIXO\n\n"
            "Responda em português brasileiro, de forma estruturada e técnica."
        )

        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.2, "maxOutputTokens": 4096},
        }
        url = f"{GEMINI_ENDPOINT}?key={self.api_key}"

        print(f"  → Gemini chamada {self.calls_made + 1}/{MAX_GEMINI_CALLS}: '{error_msg[:50]}'")
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
            print(f"  ✗ Falha Gemini: {exc}")
            return f"(erro Gemini: {exc})"


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

PRIORITY_LABELS = {"0": "Não classificado", "1": "Informação", "2": "Aviso",
                   "3": "Média", "4": "Alta", "5": "Desastre"}
VALUE_LABELS = {"0": "OK", "1": "PROBLEM"}
URGENCY_COLOR = {"CRÍTICO": "#c0392b", "ALTO": "#e67e22",
                 "MÉDIO": "#f1c40f", "BAIXO": "#27ae60", "DESCONHECIDA": "#95a5a6"}


def fmt_ts(ts_str):
    try:
        ts = int(ts_str)
        if ts == 0:
            return "nunca"
        return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    except (ValueError, TypeError):
        return ts_str or "desconhecido"


def is_old(lastclock_str):
    try:
        ts = int(lastclock_str)
        return ts == 0 or (time.time() - ts) > UNSUPPORTED_OLD_DAYS * 86400
    except (ValueError, TypeError):
        return False


def normalize_error(error):
    msg = (error or "Erro desconhecido").strip()
    msg = re.sub(r"\b\d{1,3}(\.\d{1,3}){3}\b", "<IP>", msg)
    msg = re.sub(r":\d{2,5}\b", ":<PORT>", msg)
    msg = re.sub(r"\.\d+(\.\d+){3,}", ".<OID>", msg)
    msg = re.sub(r'"[^"]{20,}"', '"<VALUE>"', msg)
    return msg


def group_items_by_error(items):
    groups = {}
    for item in items:
        raw = (item.get("error") or "Erro desconhecido").strip()
        key = normalize_error(raw)
        if key not in groups:
            groups[key] = {"raw_error": raw, "items": []}
        groups[key]["items"].append(item)
    return groups


def build_trigger_index(triggers):
    index = {}
    for trig in triggers:
        for it in trig.get("items", []):
            iid = it.get("itemid")
            if iid:
                index.setdefault(iid, []).append(trig)
    return index


def extract_urgency(analysis):
    for level in ["CRÍTICO", "ALTO", "MÉDIO", "BAIXO"]:
        if level in analysis.upper():
            return level
    return "DESCONHECIDA"


def hostname(item):
    hosts = item.get("hosts", [{}])
    return hosts[0].get("name") or hosts[0].get("host") or "Desconhecido"


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def generate_markdown(items, trigger_index, error_groups, group_analyses, changes):
    now = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    unique_hosts = len({(i.get("hosts") or [{}])[0].get("hostid") for i in items})
    blind = sum(len(v) for v in trigger_index.values())
    active = sum(1 for ts in trigger_index.values()
                 for t in ts if str(t.get("value")) == "1")

    item_to_group = {i["itemid"]: k for k, g in error_groups.items() for i in g["items"]}

    def analysis_for(item):
        return group_analyses.get(item_to_group.get(item["itemid"], ""), "")

    def trigger_table(trigs):
        if not trigs:
            return "_Nenhuma trigger._\n"
        rows = ["| Trigger | Severidade | Estado |", "|---------|-----------|--------|"]
        for t in trigs[:10]:
            val = "PROBLEM 🔴" if str(t.get("value")) == "1" else "OK ✅"
            rows.append(f"| {t.get('description','—')} | "
                        f"{PRIORITY_LABELS.get(str(t.get('priority','0')),'?')} | {val} |")
        return "\n".join(rows) + "\n"

    def item_block(item):
        hn = hostname(item)
        old = " ⚠️ ANTIGO" if is_old(item.get("lastclock", "0")) else ""
        trigs = trigger_index.get(item["itemid"], [])
        analysis = analysis_for(item)
        return (
            f"\n### HOST: {hn}\n"
            f"**Item:** {item.get('name','—')}  \n"
            f"**Key:** `{item.get('key_','—')}`  \n"
            f"**Erro:** `{item.get('error','—')}`  \n"
            f"**Última tentativa:** {fmt_ts(item.get('lastclock','0'))}{old}  \n"
            f"**Triggers:**\n\n{trigger_table(trigs)}\n"
            f"**🤖 Diagnóstico Gemini:**\n"
            f"{analysis or '_Sem análise de IA._'}\n\n---"
        )

    lines = [
        "# RELATÓRIO DE ITENS NÃO SUPORTADOS — ZABBIX",
        f"**Gerado em:** {now}  ",
        "**Analisado por:** Google Gemini 2.5 Flash  ",
        f"**Total:** {len(items)} itens | **Hosts:** {unique_hosts} | "
        f"**Grupos de erro:** {len(error_groups)}  ",
        f"**Triggers cegas:** {blind} | **Triggers em PROBLEM:** {active}  ",
        "",
    ]

    if changes["new"]:
        lines += [f"## 🆕 NOVOS ITENS SEM SUPORTE ({len(changes['new'])})", ""]
        for item in changes["new"]:
            lines.append(item_block(item))

    if changes["recovered"]:
        lines += [f"", f"## ✅ ITENS RECUPERADOS ({len(changes['recovered'])})", ""]
        for item in changes["recovered"]:
            hn = item.get("host", "?")
            lines.append(f"- **{hn}** — {item.get('name','?')} (`{item.get('key_','?')}`)")

    active_problem_ids = {
        iid for iid, ts in trigger_index.items()
        if any(str(t.get("value")) == "1" for t in ts)
    }
    critical = [i for i in items if i["itemid"] in active_problem_ids]
    alerts = [i for i in items if i["itemid"] not in active_problem_ids
              and (i.get("hosts") or [{}])[0].get("status") == "0"]
    offline = [i for i in items if (i.get("hosts") or [{}])[0].get("status") != "0"]

    lines += ["", f"## 🔴 CRÍTICOS — Triggers em PROBLEM ({len(critical)})", ""]
    if critical:
        for item in critical:
            lines.append(item_block(item))
    else:
        lines.append("_Nenhum._\n")

    lines += ["", f"## 🟡 ALERTAS — Triggers OK ({len(alerts)})", ""]
    LIMIT = 100
    for item in alerts[:LIMIT]:
        lines.append(item_block(item))
    if len(alerts) > LIMIT:
        lines.append(f"\n> {len(alerts) - LIMIT} itens adicionais — ver tabela de resumo.\n")

    if offline:
        lines += ["", f"## 🔌 HOSTS OFFLINE ({len(offline)})", ""]
        for item in offline[:50]:
            lines.append(item_block(item))

    lines += ["", "## 📊 RESUMO POR GRUPO DE ERRO", ""]
    lines.append("| Erro (normalizado) | Itens | Hosts | Urgência |")
    lines.append("|--------------------|-------|-------|----------|")
    for k, g in sorted(error_groups.items(), key=lambda x: -len(x[1]["items"])):
        h = len({(i.get("hosts") or [{}])[0].get("hostid") for i in g["items"]})
        u = extract_urgency(group_analyses.get(k, "")) if group_analyses.get(k) else "—"
        lines.append(f"| {k[:70]} | {len(g['items'])} | {h} | {u} |")

    lines += ["", "## ✅ PLANO DE AÇÃO (por urgência)", ""]
    if group_analyses:
        order = {"CRÍTICO": 0, "ALTO": 1, "MÉDIO": 2, "BAIXO": 3, "DESCONHECIDA": 4}
        for k, analysis in sorted(group_analyses.items(),
                                   key=lambda x: order.get(extract_urgency(x[1]), 4)):
            count = len(error_groups.get(k, {}).get("items", []))
            u = extract_urgency(analysis)
            lines.append(f"### [{u}] `{k[:80]}` — {count} item(s)\n")
            lines.append(analysis)
            lines.append("")
    else:
        lines.append("_Análise de IA não disponível._")

    lines += ["", "---",
              f"_Gerado por zabbix_diagnostic.py | "
              f"{len(error_groups)} grupos | {len(items)} itens | {unique_hosts} hosts_"]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------

def generate_html(items, trigger_index, error_groups, group_analyses, changes):
    now = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    unique_hosts = len({(i.get("hosts") or [{}])[0].get("hostid") for i in items})
    blind = sum(len(v) for v in trigger_index.values())
    active_problem_count = sum(1 for ts in trigger_index.values()
                                for t in ts if str(t.get("value")) == "1")
    active_problem_ids = {
        iid for iid, ts in trigger_index.items()
        if any(str(t.get("value")) == "1" for t in ts)
    }
    critical = [i for i in items if i["itemid"] in active_problem_ids]

    item_to_group = {i["itemid"]: k for k, g in error_groups.items() for i in g["items"]}

    def urgency_badge(u):
        color = URGENCY_COLOR.get(u, "#95a5a6")
        return f'<span style="background:{color};color:#fff;padding:2px 8px;border-radius:3px;font-size:12px;font-weight:bold">{u}</span>'

    def group_summary_rows():
        rows = []
        for k, g in sorted(error_groups.items(), key=lambda x: -len(x[1]["items"])):
            h = len({(i.get("hosts") or [{}])[0].get("hostid") for i in g["items"]})
            analysis = group_analyses.get(k, "")
            u = extract_urgency(analysis) if analysis else "—"
            badge = urgency_badge(u) if analysis else u
            rows.append(
                f"<tr>"
                f"<td style='font-family:monospace;font-size:12px'>{k[:80]}</td>"
                f"<td style='text-align:center'><strong>{len(g['items'])}</strong></td>"
                f"<td style='text-align:center'>{h}</td>"
                f"<td style='text-align:center'>{badge}</td>"
                f"</tr>"
            )
        return "\n".join(rows)

    def change_section():
        if not changes["new"] and not changes["recovered"]:
            return ""
        parts = []
        if changes["new"]:
            items_html = "".join(
                f"<li><strong>{hostname(i)}</strong> — {i.get('name','?')}<br>"
                f"<code>{i.get('error','?')}</code></li>"
                for i in changes["new"]
            )
            parts.append(
                f'<div class="card new">'
                f'<h3>🆕 Novos Itens Sem Suporte ({len(changes["new"])})</h3>'
                f'<ul>{items_html}</ul></div>'
            )
        if changes["recovered"]:
            items_html = "".join(
                f"<li><strong>{i.get('host','?')}</strong> — {i.get('name','?')}</li>"
                for i in changes["recovered"]
            )
            parts.append(
                f'<div class="card recovered">'
                f'<h3>✅ Itens Recuperados ({len(changes["recovered"])})</h3>'
                f'<ul>{items_html}</ul></div>'
            )
        return "\n".join(parts)

    def action_plan_html():
        if not group_analyses:
            return "<p><em>Análise de IA não disponível.</em></p>"
        order = {"CRÍTICO": 0, "ALTO": 1, "MÉDIO": 2, "BAIXO": 3, "DESCONHECIDA": 4}
        parts = []
        for k, analysis in sorted(group_analyses.items(),
                                   key=lambda x: order.get(extract_urgency(x[1]), 4)):
            count = len(error_groups.get(k, {}).get("items", []))
            u = extract_urgency(analysis)
            color = URGENCY_COLOR.get(u, "#95a5a6")
            # Convert markdown-ish to basic HTML
            html_analysis = analysis.replace("\n", "<br>").replace("**", "<strong>", 1)
            parts.append(
                f'<div style="border-left:4px solid {color};padding:12px 16px;'
                f'margin-bottom:16px;background:#fafafa;border-radius:0 4px 4px 0">'
                f'<strong style="color:{color}">[{u}]</strong> '
                f'<code style="font-size:12px">{k[:80]}</code> — {count} item(s)<br><br>'
                f'<div style="font-size:13px;line-height:1.6">{analysis.replace(chr(10), "<br>")}</div>'
                f'</div>'
            )
        return "\n".join(parts)

    html = f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Zabbix — Itens Não Suportados</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
          background: #f0f2f5; color: #333; }}
  .header {{ background: #d40000; color: white; padding: 24px 32px; }}
  .header h1 {{ font-size: 22px; font-weight: 700; }}
  .header .sub {{ font-size: 13px; opacity: .85; margin-top: 4px; }}
  .content {{ max-width: 1200px; margin: 0 auto; padding: 24px 16px; }}
  .kpi-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px,1fr));
               gap: 16px; margin-bottom: 24px; }}
  .kpi {{ background: white; border-radius: 8px; padding: 20px; text-align: center;
          box-shadow: 0 1px 4px rgba(0,0,0,.08); }}
  .kpi .val {{ font-size: 36px; font-weight: 800; }}
  .kpi .lbl {{ font-size: 12px; color: #666; margin-top: 4px; text-transform: uppercase; }}
  .kpi.critical .val {{ color: #c0392b; }}
  .kpi.warn .val {{ color: #e67e22; }}
  .kpi.ok .val {{ color: #27ae60; }}
  .card {{ background: white; border-radius: 8px; padding: 24px;
           box-shadow: 0 1px 4px rgba(0,0,0,.08); margin-bottom: 20px; }}
  .card h2 {{ font-size: 16px; margin-bottom: 16px; border-bottom: 2px solid #f0f2f5;
              padding-bottom: 10px; }}
  .card h3 {{ font-size: 14px; margin-bottom: 10px; }}
  .card.new {{ border-left: 4px solid #e74c3c; }}
  .card.recovered {{ border-left: 4px solid #27ae60; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  th {{ background: #f7f8fa; text-align: left; padding: 8px 12px;
        border-bottom: 2px solid #e0e0e0; font-size: 12px; text-transform: uppercase; color: #666; }}
  td {{ padding: 8px 12px; border-bottom: 1px solid #f0f0f0; vertical-align: top; }}
  tr:hover td {{ background: #fafbfc; }}
  code {{ background: #f0f2f5; padding: 2px 6px; border-radius: 3px; font-size: 12px; }}
  ul {{ padding-left: 20px; }}
  li {{ margin-bottom: 6px; line-height: 1.5; }}
  .footer {{ text-align: center; color: #999; font-size: 12px; padding: 32px; }}
</style>
</head>
<body>
<div class="header">
  <h1>🔍 Zabbix — Relatório de Itens Não Suportados</h1>
  <div class="sub">Gerado em {now} · Analisado por Google Gemini 2.5 Flash</div>
</div>
<div class="content">

  <div class="kpi-grid">
    <div class="kpi critical">
      <div class="val">{len(items)}</div>
      <div class="lbl">Itens sem suporte</div>
    </div>
    <div class="kpi warn">
      <div class="val">{unique_hosts}</div>
      <div class="lbl">Hosts afetados</div>
    </div>
    <div class="kpi warn">
      <div class="val">{len(error_groups)}</div>
      <div class="lbl">Grupos de erro</div>
    </div>
    <div class="kpi critical">
      <div class="val">{blind}</div>
      <div class="lbl">Triggers cegas</div>
    </div>
    <div class="kpi critical">
      <div class="val">{active_problem_count}</div>
      <div class="lbl">PROBLEM ativo</div>
    </div>
    <div class="kpi {'critical' if changes['new'] else 'ok'}">
      <div class="val">{len(changes['new'])}</div>
      <div class="lbl">Novos (esta varredura)</div>
    </div>
    <div class="kpi ok">
      <div class="val">{len(changes['recovered'])}</div>
      <div class="lbl">Recuperados</div>
    </div>
  </div>

  {change_section()}

  <div class="card">
    <h2>📊 Resumo por Grupo de Erro</h2>
    <table>
      <tr>
        <th>Erro (normalizado)</th>
        <th style="text-align:center">Itens</th>
        <th style="text-align:center">Hosts</th>
        <th style="text-align:center">Urgência</th>
      </tr>
      {group_summary_rows()}
    </table>
  </div>

  <div class="card">
    <h2>✅ Plano de Ação Consolidado (IA)</h2>
    {action_plan_html()}
  </div>

  <div class="card">
    <h2>🔴 Itens Críticos ({len(critical)}) — Triggers em PROBLEM</h2>
    {'<table><tr><th>Host</th><th>Item</th><th>Erro</th><th>Última tentativa</th></tr>' +
     ''.join(f'<tr><td>{hostname(i)}</td><td>{i.get("name","—")}<br><code>{i.get("key_","")}</code></td>'
             f'<td><code>{i.get("error","—")}</code></td><td>{fmt_ts(i.get("lastclock","0"))}</td></tr>'
             for i in critical) + '</table>'
     if critical else '<p><em>Nenhum item crítico.</em></p>'}
  </div>

</div>
<div class="footer">
  Relatório gerado por zabbix_diagnostic.py ·
  {len(error_groups)} grupos · {len(items)} itens · {unique_hosts} hosts
</div>
</body>
</html>"""
    return html


# ---------------------------------------------------------------------------
# Email
# ---------------------------------------------------------------------------

def send_email(env: dict, subject: str, body_html: str, attachments: list):
    """Send HTML email with optional file attachments."""
    if not env["SMTP_HOST"] or not env["ALERT_EMAIL"]:
        print("  ℹ Email não configurado — relatório salvo localmente apenas.")
        return

    recipients = [r.strip() for r in env["ALERT_EMAIL"].split(",") if r.strip()]
    msg = MIMEMultipart("mixed")
    msg["From"] = env["SMTP_USER"]
    msg["To"] = ", ".join(recipients)
    msg["Subject"] = subject

    # HTML body
    body = MIMEMultipart("alternative")
    body.attach(MIMEText(body_html, "html", "utf-8"))
    msg.attach(body)

    # Attachments
    for path in attachments:
        p = Path(path)
        if not p.exists():
            continue
        part = MIMEBase("application", "octet-stream")
        part.set_payload(p.read_bytes())
        encoders.encode_base64(part)
        part.add_header("Content-Disposition", f'attachment; filename="{p.name}"')
        msg.attach(part)

    print(f"→ Enviando e-mail para {recipients}...")
    try:
        with smtplib.SMTP(env["SMTP_HOST"], env["SMTP_PORT"]) as smtp:
            smtp.ehlo()
            smtp.starttls()
            smtp.login(env["SMTP_USER"], env["SMTP_PASSWORD"])
            smtp.sendmail(env["SMTP_USER"], recipients, msg.as_bytes())
        print("  ✓ E-mail enviado.")
    except Exception as exc:
        print(f"  ✗ Falha ao enviar e-mail: {exc}")


def build_email_html(items, changes, error_groups, group_analyses, unique_hosts):
    """Compact HTML suitable for email body (no external resources)."""
    now = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    new_count = len(changes["new"])
    rec_count = len(changes["recovered"])
    groups_count = len(error_groups)

    def urgency_badge(u):
        color = URGENCY_COLOR.get(u, "#95a5a6")
        return (f'<span style="background:{color};color:#fff;padding:1px 6px;'
                f'border-radius:3px;font-size:11px">{u}</span>')

    new_rows = "".join(
        f"<tr>"
        f"<td style='padding:6px 8px'><strong>{hostname(i)}</strong></td>"
        f"<td style='padding:6px 8px'>{i.get('name','?')}</td>"
        f"<td style='padding:6px 8px;font-family:monospace;font-size:11px'>"
        f"{i.get('error','?')[:80]}</td>"
        f"</tr>"
        for i in changes["new"][:20]
    )

    rec_rows = "".join(
        f"<tr><td style='padding:6px 8px'>{i.get('host','?')}</td>"
        f"<td style='padding:6px 8px'>{i.get('name','?')}</td></tr>"
        for i in changes["recovered"][:10]
    )

    group_rows = "".join(
        f"<tr>"
        f"<td style='padding:6px 8px;font-family:monospace;font-size:11px'>{k[:60]}</td>"
        f"<td style='padding:6px 8px;text-align:center'>{len(g['items'])}</td>"
        f"<td style='padding:6px 8px;text-align:center'>"
        f"{urgency_badge(extract_urgency(group_analyses.get(k,''))) if group_analyses.get(k) else '—'}"
        f"</td>"
        f"</tr>"
        for k, g in sorted(error_groups.items(), key=lambda x: -len(x[1]["items"]))[:10]
    )

    return f"""
<div style="font-family:Arial,sans-serif;max-width:700px;margin:0 auto">
  <div style="background:#d40000;color:#fff;padding:16px 20px;border-radius:6px 6px 0 0">
    <h2 style="margin:0;font-size:18px">🔍 Zabbix — Alerta de Itens Não Suportados</h2>
    <p style="margin:4px 0 0;font-size:12px;opacity:.85">{now}</p>
  </div>
  <div style="background:#fff;padding:20px;border:1px solid #ddd;border-top:none">

    <table style="width:100%;border-collapse:collapse;margin-bottom:20px">
      <tr>
        <td style="text-align:center;padding:12px;background:#fdf2f2;border-radius:6px">
          <div style="font-size:28px;font-weight:800;color:#c0392b">{len(items)}</div>
          <div style="font-size:11px;color:#666">Total sem suporte</div>
        </td>
        <td style="text-align:center;padding:12px;background:#fef9e7;border-radius:6px">
          <div style="font-size:28px;font-weight:800;color:#e67e22">{unique_hosts}</div>
          <div style="font-size:11px;color:#666">Hosts afetados</div>
        </td>
        <td style="text-align:center;padding:12px;background:#fdf2f2;border-radius:6px">
          <div style="font-size:28px;font-weight:800;color:#c0392b">{new_count}</div>
          <div style="font-size:11px;color:#666">🆕 Novos</div>
        </td>
        <td style="text-align:center;padding:12px;background:#eafaf1;border-radius:6px">
          <div style="font-size:28px;font-weight:800;color:#27ae60">{rec_count}</div>
          <div style="font-size:11px;color:#666">✅ Recuperados</div>
        </td>
      </tr>
    </table>

    {"" if not changes["new"] else f'''
    <h3 style="color:#c0392b;margin:0 0 8px">🆕 Novos Itens Sem Suporte</h3>
    <table style="width:100%;border-collapse:collapse;font-size:13px;margin-bottom:20px">
      <tr style="background:#f7f8fa">
        <th style="padding:6px 8px;text-align:left">Host</th>
        <th style="padding:6px 8px;text-align:left">Item</th>
        <th style="padding:6px 8px;text-align:left">Erro</th>
      </tr>
      {new_rows}
    </table>
    {"<p style='font-size:12px;color:#999'>... e mais itens — ver relatório completo em anexo.</p>" if new_count > 20 else ""}
    '''}

    {"" if not changes["recovered"] else f'''
    <h3 style="color:#27ae60;margin:0 0 8px">✅ Itens Recuperados</h3>
    <table style="width:100%;border-collapse:collapse;font-size:13px;margin-bottom:20px">
      <tr style="background:#f7f8fa">
        <th style="padding:6px 8px;text-align:left">Host</th>
        <th style="padding:6px 8px;text-align:left">Item</th>
      </tr>
      {rec_rows}
    </table>
    '''}

    <h3 style="margin:0 0 8px">📊 Top Grupos de Erro</h3>
    <table style="width:100%;border-collapse:collapse;font-size:13px">
      <tr style="background:#f7f8fa">
        <th style="padding:6px 8px;text-align:left">Erro</th>
        <th style="padding:6px 8px;text-align:center">Itens</th>
        <th style="padding:6px 8px;text-align:center">Urgência</th>
      </tr>
      {group_rows}
    </table>

    <p style="font-size:12px;color:#999;margin-top:20px">
      Relatório completo em HTML e Markdown em anexo.
      Gerado por zabbix_diagnostic.py.
    </p>
  </div>
</div>"""


# ---------------------------------------------------------------------------
# Main logic (single run)
# ---------------------------------------------------------------------------

def run_once(env: dict, use_gemini: bool = True, force_email: bool = False) -> dict:
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Iniciando varredura...")

    zabbix = ZabbixClient(
        url=env["ZABBIX_URL"],
        token=env["ZABBIX_TOKEN"],
        user=env["ZABBIX_USER"],
        password=env["ZABBIX_PASSWORD"],
    )
    zabbix.login()

    items = zabbix.get_unsupported_items()

    # State + change detection (for intermittent items)
    state = load_state()
    changes = detect_changes(items, state)

    if not items:
        print("✓ Nenhum item não suportado. Ambiente saudável!")
        save_state(changes["new_state"])
        if changes["recovered"] and (changes["has_changes"] or force_email):
            # Items all recovered — still worth notifying
            pass
        return changes

    print(f"\n→ Mudanças detectadas: "
          f"{len(changes['new'])} novo(s), {len(changes['recovered'])} recuperado(s), "
          f"{len(changes['persistent'])} persistente(s).")

    # Group errors
    error_groups = group_items_by_error(items)

    # Triggers for sample items only
    sample_ids = list({
        i["itemid"]
        for g in error_groups.values()
        for i in g["items"][:SAMPLE_PER_GROUP]
    })
    triggers = zabbix.get_triggers_for_items(sample_ids)
    trigger_index = build_trigger_index(triggers)

    # Gemini analysis
    group_analyses = {}
    if use_gemini:
        print(f"\n→ Analisando {len(error_groups)} grupo(s) com Gemini AI...")
        gemini = GeminiClient(env["GEMINI_API_KEY"])
        for grp_key, grp in sorted(error_groups.items(), key=lambda x: -len(x[1]["items"])):
            grp_items = grp["items"]
            sample = grp_items[:SAMPLE_PER_GROUP]
            h_count = len({(i.get("hosts") or [{}])[0].get("hostid") for i in grp_items})
            sample_ids_set = {i["itemid"] for i in sample}
            sample_triggers = list({
                t["triggerid"]: t
                for iid, ts in trigger_index.items()
                if iid in sample_ids_set
                for t in ts
            }.values())[:10]

            analysis = gemini.analyze_group(
                error_msg=grp_key,
                sample_items=sample,
                sample_triggers=sample_triggers,
                total_count=len(grp_items),
                host_count=h_count,
            )
            group_analyses[grp_key] = analysis

            if gemini.calls_made >= MAX_GEMINI_CALLS:
                remaining = len(error_groups) - len(group_analyses)
                if remaining > 0:
                    print(f"  ⚠ Limite atingido. {remaining} grupo(s) sem análise.")
                break

    # Reports
    print("\n→ Gerando relatórios...")
    unique_hosts = len({(i.get("hosts") or [{}])[0].get("hostid") for i in items})

    md = generate_markdown(items, trigger_index, error_groups, group_analyses, changes)
    REPORT_MD.write_text(md, encoding="utf-8")

    html = generate_html(items, trigger_index, error_groups, group_analyses, changes)
    REPORT_HTML.write_text(html, encoding="utf-8")

    print(f"  ✓ {REPORT_MD.name} e {REPORT_HTML.name} salvos.")

    # Email — send only when there are changes (or forced)
    if changes["has_changes"] or force_email:
        subject = (
            f"[Zabbix] {len(changes['new'])} novo(s) item(s) sem suporte"
            + (f", {len(changes['recovered'])} recuperado(s)" if changes["recovered"] else "")
            + f" — {datetime.now().strftime('%d/%m/%Y %H:%M')}"
        )
        email_html = build_email_html(items, changes, error_groups, group_analyses, unique_hosts)
        send_email(env, subject, email_html, [REPORT_HTML, REPORT_MD])
    else:
        print("  ℹ Sem mudanças desde a última varredura — e-mail não enviado.")

    # Save state after successful run
    save_state(changes["new_state"])

    # Executive summary
    print()
    print("=" * 60)
    print("RESUMO DA VARREDURA")
    print("=" * 60)
    print(f"1. {len(items)} itens sem suporte | {unique_hosts} host(s) | "
          f"{len(error_groups)} grupo(s) de erro.")
    print(f"2. Mudanças: {len(changes['new'])} novo(s) | {len(changes['recovered'])} recuperado(s) "
          f"| {len(changes['persistent'])} persistente(s).")
    active_problems = sum(1 for ts in trigger_index.values()
                          for t in ts if str(t.get("value")) == "1")
    print(f"3. {active_problems} trigger(s) em PROBLEM ativo.")
    old_count = sum(1 for i in items if is_old(i.get("lastclock", "0")))
    print(f"4. {old_count} item(s) sem suporte há mais de {UNSUPPORTED_OLD_DAYS} dias.")
    critical_groups = [k for k, v in group_analyses.items() if extract_urgency(v) == "CRÍTICO"]
    if critical_groups:
        print(f"5. ⚠ {len(critical_groups)} grupo(s) CRÍTICO(s) — ação imediata recomendada.")
    else:
        print(f"5. Relatórios: {REPORT_HTML.resolve()}")
    print("=" * 60)

    return changes


# ---------------------------------------------------------------------------
# Watch mode
# ---------------------------------------------------------------------------

def watch_mode(env: dict, interval_minutes: int, use_gemini: bool):
    print(f"\n🔄 Modo watch ativado — varredura a cada {interval_minutes} minuto(s).")
    print("   Pressione Ctrl+C para parar.\n")
    while True:
        try:
            run_once(env, use_gemini=use_gemini)
            next_run = datetime.now(tz=timezone.utc)
            print(f"\n⏳ Próxima varredura em {interval_minutes} min "
                  f"(~{next_run.strftime('%H:%M')} UTC + {interval_minutes}min).\n")
            time.sleep(interval_minutes * 60)
        except KeyboardInterrupt:
            print("\n\n✓ Watch encerrado pelo usuário.")
            break
        except Exception as exc:
            print(f"\n✗ Erro na varredura: {exc}")
            print(f"  Aguardando {interval_minutes} min antes de tentar novamente...")
            time.sleep(interval_minutes * 60)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Zabbix Unsupported Items Diagnostic Agent + Gemini AI"
    )
    parser.add_argument(
        "--watch", metavar="MINUTOS", type=int, default=0,
        help="Rodar em loop a cada N minutos (ex: --watch 10)"
    )
    parser.add_argument(
        "--no-gemini", action="store_true",
        help="Desativar análise Gemini (mais rápido, sem consumo de quota)"
    )
    parser.add_argument(
        "--force-email", action="store_true",
        help="Enviar e-mail mesmo sem mudanças (útil para teste)"
    )
    args = parser.parse_args()

    print()
    print("=" * 60)
    print("  ZABBIX DIAGNOSTIC AGENT — Powered by Google Gemini")
    print("=" * 60)

    env = load_env()

    if args.watch:
        watch_mode(env, interval_minutes=args.watch, use_gemini=not args.no_gemini)
    else:
        run_once(env, use_gemini=not args.no_gemini, force_email=args.force_email)


if __name__ == "__main__":
    main()
