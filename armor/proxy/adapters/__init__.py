from armor.proxy.adapters.anthropic import AnthropicMessagesAdapter
from armor.proxy.adapters.openai import OpenAIChatAdapter

# ponytail: keyed by wire-shape adapter *type*, not provider name. Most
# providers in configs/proxy/providers.json share the "openai" type since
# they expose an OpenAI-compatible endpoint — this is what makes adding a
# new provider a config-only change (armor/proxy/config.py resolves each
# provider to one of these type names via UpstreamConfig.adapter).
ADAPTERS = {
    "openai": OpenAIChatAdapter(),
    "anthropic": AnthropicMessagesAdapter(),
}


def get_adapter(adapter_type: str):
    return ADAPTERS[adapter_type]
