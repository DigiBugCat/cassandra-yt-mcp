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

variable "worker_script_name" {
  description = "Worker script name deployed by Wrangler"
  type        = string
}

variable "worker_subdomain" {
  description = "Public MCP Worker subdomain"
  type        = string
}

variable "enable_waf_skip" {
  description = "Whether to skip managed WAF rules for the MCP hostname"
  type        = bool
  default     = true
}
