terraform {
  required_providers {
    cloudflare = {
      source  = "cloudflare/cloudflare"
      version = "~> 4.0"
    }
  }
}

resource "cloudflare_workers_kv_namespace" "oauth" {
  account_id = var.account_id
  title      = "${var.worker_script_name}-oauth-state"
}

resource "cloudflare_record" "worker" {
  zone_id = var.zone_id
  name    = var.worker_subdomain
  content = "${var.worker_script_name}.${var.account_id}.workers.dev"
  type    = "CNAME"
  proxied = true
  comment = "Public Cassandra YT MCP worker hostname"
}

resource "cloudflare_ruleset" "mcp_waf_skip" {
  count   = var.enable_waf_skip ? 1 : 0
  zone_id = var.zone_id
  name    = "Skip bot rules for ${var.worker_subdomain}"
  kind    = "zone"
  phase   = "http_request_firewall_managed"

  rules {
    action      = "skip"
    expression  = "(http.host eq \"${var.worker_subdomain}.${var.domain}\")"
    description = "Allow MCP clients through managed bot rules"

    action_parameters {
      ruleset = "current"
    }
  }
}
