# Verified *_BASE_URL environment variables actually read by each
# provider's official SDK at client-construction time. Only add an entry
# here after confirming it in that SDK's own docs/source — a wrong guess
# here would silently fail to redirect traffic, worse than showing nothing.
# openai: https://claude4u.com/en/guide/openai-base-url-config
# anthropic: https://github.com/anthropics/anthropic-sdk-python/issues/985
KNOWN_ENV_OVERRIDES = {
    "openai": "OPENAI_BASE_URL",
    "anthropic": "ANTHROPIC_BASE_URL",
}
