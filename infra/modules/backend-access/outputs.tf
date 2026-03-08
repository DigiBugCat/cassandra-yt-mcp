output "backend_hostname" {
  description = "Protected backend hostname"
  value       = "${var.backend_subdomain}.${var.domain}"
}

output "access_application_id" {
  description = "Cloudflare Access application ID"
  value       = cloudflare_zero_trust_access_application.backend.id
}

output "cf_access_client_id" {
  description = "Service token client ID for Worker backend requests"
  value       = cloudflare_zero_trust_access_service_token.backend.client_id
}

output "cf_access_client_secret" {
  description = "Service token client secret for Worker backend requests"
  value       = cloudflare_zero_trust_access_service_token.backend.client_secret
  sensitive   = true
}
