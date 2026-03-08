variable "account_id" {
  description = "Cloudflare account ID"
  type        = string
}

variable "zone_id" {
  description = "Cloudflare zone ID"
  type        = string
}

variable "domain" {
  description = "Root domain name"
  type        = string
}

variable "backend_subdomain" {
  description = "Private backend hostname subdomain"
  type        = string
}

variable "application_name" {
  description = "Access application name"
  type        = string
  default     = "cassandra-yt-mcp-backend"
}
