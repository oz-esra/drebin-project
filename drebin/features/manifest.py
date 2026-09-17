"""Feature sets S1-S4 from AndroidManifest.xml (Drebin, Arp et al. NDSS 2014, Sec. II-A.1)."""

from androguard.core.apk import APK

# One prefix per feature set, as the paper requires: elements of different
# sets must not collide in the joint set S.
S1_HARDWARE = "feature"
S2_PERMISSION = "permission"
S3_COMPONENT = "component"
S4_INTENT = "intent"

_COMPONENT_KINDS = {
    "activity": "get_activities",
    "service": "get_services",
    "receiver": "get_receivers",
    "provider": "get_providers",
}


def hardware_components(apk: APK) -> list[str]:
    """S1: uses-feature entries."""
    return sorted(apk.get_features())


def requested_permissions(apk: APK) -> list[str]:
    """S2: uses-permission entries."""
    return sorted(set(apk.get_permissions()))


def app_components(apk: APK) -> list[str]:
    """S3: declared activities, services, receivers and providers."""
    names = set()
    for getter in _COMPONENT_KINDS.values():
        names.update(getattr(apk, getter)() or [])
    return sorted(names)


def filtered_intents(apk: APK) -> list[str]:
    """S4: intent-filter actions declared by any component."""
    actions = set()
    for kind, getter in _COMPONENT_KINDS.items():
        for name in getattr(apk, getter)() or []:
            filters = apk.get_intent_filters(kind, name) or {}
            actions.update(filters.get("action", []))
    return sorted(actions)


def extract(apk: APK) -> dict[str, list[str]]:
    """Return S1-S4 keyed by their prefix."""
    return {
        S1_HARDWARE: hardware_components(apk),
        S2_PERMISSION: requested_permissions(apk),
        S3_COMPONENT: app_components(apk),
        S4_INTENT: filtered_intents(apk),
    }