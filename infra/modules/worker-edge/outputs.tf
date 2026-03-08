output "kv_namespace_id" {
  description = "KV namespace ID to bind in wrangler.jsonc"
  value       = cloudflare_workers_kv_namespace.oauth.id
}

output "worker_hostname" {
  description = "Public Worker hostname"
  value       = "${var.worker_subdomain}.${var.domain}"
}

output "mcp_url" {
  description = "Public MCP URL"
  value       = "https://${var.worker_subdomain}.${var.domain}/mcp"
}

output "callback_url" {
  description = "WorkOS redirect URI"
  value       = "https://${var.worker_subdomain}.${var.domain}/callback"
}
