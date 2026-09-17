"""Extract S5-S8 features from DEX bytecode (Androguard 4.x Compatible)."""

import ipaddress
import re

from androguard.core.analysis.analysis import Analysis
from androguard.core.apk import APK
from androguard.core.dex import DEX

# =============================================================================
# Architecture Strategy (S5 - S8 Feature Extraction):
# 1. Feature Keys & Target API: Align with Drebin dataset standards (API Level 16).
# 2. Rule Definitions: Method signatures, system commands, TLDs, and network regexes.
# 3. Disambiguation Helpers: Filter Java package false positives and reserved IPs.
# 4. Permission Mapping (S5/S6): Correlate API calls with requested manifest permissions.
# 5. Multi-DEX Scanning (S7/S8): Process method IDs and string pools across DEX files.
# =============================================================================

# Drebin official feature prefixes
S5_RESTRICTED_API = "api_call"
S6_USED_PERMISSION = "real_permission"
S7_SUSPICIOUS_API = "call"
S8_NETWORK_ADDRESS = "url"

# Fixed Android API level used for mapping API calls to permissions.
# Pinned to Level 16 (Android 4.1, typical for the Drebin dataset era 2010-2012)
# to avoid comparative variance across samples built against different target SDKs.
API_LEVEL = 16

# S7: Suspicious method call patterns matched against bytecode method references
SUSPICIOUS_METHOD_PATTERNS = [
    "getDeviceId",
    "getSubscriberId",
    "sendTextMessage",
    "sendMultipartTextMessage",
    "execHttpRequest",
    "setWifiEnabled",
    "Ljava/lang/Runtime;->exec",
    "Ljava/lang/reflect",
    "Ldalvik/system/DexClassLoader",
    "Ljavax/crypto/Cipher",
]

# S7: Suspicious string constants embedded in DEX string pools.
# Captures shell commands, root binaries, and execution paths that cannot
# be detected solely through method invocation parsing.
SUSPICIOUS_STRING_PATTERNS = [
    "system/bin/su",
    "system/xbin/su",
    "/system/bin/sh",
    "chmod",
    "chown",
    "mount -o",
    "busybox",
    "pm install",
    "su -c",
]

# S8: Permitted Top-Level Domains (TLDs) for scheme-less hostname extraction
TLDS = (
    "com", "net", "org", "info", "biz", "ru", "cn", "co", "io", "me", "tk",
    "xyz", "top", "club", "online", "site", "app", "de", "uk", "fr", "nl",
    "pl", "br", "in", "jp", "kr", "us", "gg",
)

# S8: Full URL regex capturing protocol, path, and query parameters.
# Note: Retains XML namespaces (e.g., http://schemas.android.com/apk/res/android);
# linear classifiers naturally assign near-zero weight to these ubiquitous noise features.
URL_REGEX = re.compile(
    r"https?://(?:[a-zA-Z0-9$\-_@.&+/?:#=]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
)

# S8: Context-bound IPv4 regex requiring scheme (`//`), userinfo (`@`), or explicit port suffix.
# Prevents false positive extraction of version strings and X.509 Object Identifiers (e.g., 1.3.6.1).
_IP_OCTET = r"(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)"
_IP_BODY = rf"{_IP_OCTET}\.{_IP_OCTET}\.{_IP_OCTET}\.{_IP_OCTET}"

IP_REGEX = re.compile(
    r"(?:(?<=//)|(?<=@))"          # Preceded by scheme or userinfo
    rf"({_IP_BODY})"
    r"|"
    rf"\b({_IP_BODY})(?=:\d)"      # Followed by explicit port
)

# S8: Scheme-less hostname regex.
# Excludes boundary extensions to avoid slicing intent identifiers (e.g., "action.de.danoeh.antennapod").
# Case-sensitive matching avoids picking up system constants like "Log.INFO" or "Dispatchers.IO".
HOST_REGEX = re.compile(
    r"\b(?:[a-z0-9](?:[a-z0-9\-]{0,61}[a-z0-9])?\.)+"
    r"(?:" + "|".join(TLDS) + r")(?![a-z0-9\-.])"
)

# Standard framework package prefixes to filter out during hostname extraction
PACKAGE_PREFIXES = (
    "android.",
    "androidx.",
    "java.",
    "javax.",
    "kotlin.",
    "kotlinx.",
    "dalvik.",
    "sun.",
    "junit.",
)


def method_signature(class_name: str, method_name: str) -> str:
    """Normalize class and method identifiers into a unified canonical signature.

    Centralizing formatting prevents representation divergence between S5 and S7 sets.
    """
    return f"{class_name}->{method_name}"


def looks_like_package(candidate: str) -> bool:
    """Disambiguate hostnames from Java package names using structural heuristics.

    Evaluates reverse domain naming conventions (packages start with TLDs, hostnames end with TLDs)
    and strips known application/framework package prefixes.
    """
    low = candidate.lower()
    if low.startswith(PACKAGE_PREFIXES):
        return True
    return low.split(".")[0] in TLDS


def is_valid_ip(ip_str: str) -> bool:
    """Validate IP syntax and filter non-routable or local network blocks."""
    try:
        ip = ipaddress.ip_address(ip_str)
    except ValueError:
        return False
    return not (
        ip.is_unspecified
        or ip.is_loopback
        or ip.is_multicast
        or ip.is_link_local
        or ip.is_reserved
    )


def permission_mapped_calls(
    apk: APK, dx: Analysis, api_level: int = API_LEVEL
) -> tuple[set[str], set[str]]:
    """Extract restricted API invocations (S5) and active requested permissions (S6).

    - S5: Permission-protected API calls detected in bytecode.
    - S6: Intersection between implied API permissions and manifest-requested permissions.
    """
    calls = set()
    implied = set()

    for method_analysis, permissions in dx.get_permissions(api_level):
        m = method_analysis.get_method()
        calls.add(method_signature(m.get_class_name(), m.get_name()))
        implied.update(permissions)

    requested = set(apk.get_permissions())
    return calls, implied & requested


def extract(apk: APK, dx: Analysis) -> dict[str, list[str]]:
    """Traverse all DEX structures to extract and categorize S5-S8 features."""
    # Retrieve S5 and S6 via Androguard permission mapping analysis
    s5_restricted, s6_used_perms = permission_mapped_calls(apk, dx)

    s7_suspicious = set()
    s8_network = set()

    # Iterate through all available DEX bytecodes (handles multi-DEX applications)
    for dex_bytes in apk.get_all_dex():
        dvm = DEX(dex_bytes)

        # 1. Analyze Bytecode Method References (S7)
        methods_item = dvm.get_methods_id_item()
        if methods_item:
            for m in methods_item.get_obj():
                full_sig = method_signature(m.get_class_name(), m.get_name())
                for pattern in SUSPICIOUS_METHOD_PATTERNS:
                    if pattern in full_sig:
                        s7_suspicious.add(full_sig)

        # 2. Analyze String Pool (S7 String Constants + S8 Network Indicators)
        strings = dvm.get_strings()
        if strings:
            for string_val in strings:
                # S7: Record explicit pattern matches for shell/root commands
                for pattern in SUSPICIOUS_STRING_PATTERNS:
                    if pattern in string_val:
                        s7_suspicious.add(pattern)

                # S8: Extract URLs (including query strings and paths)
                for url in URL_REGEX.findall(string_val):
                    s8_network.add(url)

                # S8: Extract validated IPv4 addresses in network context
                for match in IP_REGEX.findall(string_val):
                    ip = next((g for g in match if g), None)
                    if ip and is_valid_ip(ip):
                        s8_network.add(ip)

                # S8: Extract bare hostnames with package filtering
                for host in HOST_REGEX.findall(string_val):
                    if not looks_like_package(host):
                        s8_network.add(host)

    return {
        S5_RESTRICTED_API: sorted(s5_restricted),
        S6_USED_PERMISSION: sorted(s6_used_perms),
        S7_SUSPICIOUS_API: sorted(s7_suspicious),
        S8_NETWORK_ADDRESS: sorted(s8_network),
    }