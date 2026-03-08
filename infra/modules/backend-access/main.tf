terraform {
  required_providers {
    cloudflare = {
      source  = "cloudflare/cloudflare"
      version = "~> 4.0"
    }
  }
}

resource "cloudflare_zero_trust_access_application" "backend" {
  zone_id                    = var.zone_id
  name                       = var.application_name
  domain                     = "${var.backend_subdomain}.${var.domain}"
  type                       = "self_hosted"
  session_duration           = "24h"
  auto_redirect_to_identity  = false
  http_only_cookie_attribute = false
}

resource "cloudflare_zero_trust_access_service_token" "backend" {
  account_id = var.account_id
  name       = "${var.application_name}-worker"
}

resource "cloudflare_zero_trust_access_policy" "backend" {
  application_id = cloudflare_zero_trust_access_application.backend.id
  zone_id        = var.zone_id
  name           = "Worker service token access"
  precedence     = 1
  decision       = "non_identity"

  include {
    service_token = [cloudflare_zero_trust_access_service_token.backend.id]
  }
}
