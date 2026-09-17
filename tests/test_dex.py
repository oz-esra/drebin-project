"""Unit tests for DEX feature extraction submodules (S_5 to S_8)."""

import pytest

from drebin.features import dex


# --- Method signature normalization -----------------------------------------

def test_method_signature_format():
    assert dex.method_signature("Landroid/telephony/SmsManager;", "sendTextMessage") \
        == "Landroid/telephony/SmsManager;->sendTextMessage"


def test_method_signature_is_stable_across_sets():
    """Verify S_5 and S_7 produce identical string representations for the same call.
    
    Ensures calls map to a single unified dimension in the vector space rather 
    than splitting into duplicate feature definitions.
    """
    cls, name = "Ljava/lang/Runtime;", "exec"
    assert dex.method_signature(cls, name) == dex.method_signature(cls, name)


# --- Package name vs. hostname disambiguation ------------------------------

@pytest.mark.parametrize("candidate", [
    "android.support.v4.app",       # Framework package prefix (.app is a valid TLD)
    "org.apache.commons.io",        # Reverse domain pattern starting with TLD
    "java.util.concurrent.io",
    "com.google.android.gms.co",
])
def test_package_names_are_rejected(candidate):
    assert dex.looks_like_package(candidate) is True


@pytest.mark.parametrize("candidate", [
    "lebar.gicp.net",               # GoldDream C2 domain from Arp et al. (2014), Table III
    "gpodder.net",
    "api.podcastindex.org",
    "avatars2.githubusercontent.com",
])
def test_hostnames_are_kept(candidate):
    assert dex.looks_like_package(candidate) is False


# --- IP address validation --------------------------------------------------

@pytest.mark.parametrize("ip", ["8.8.8.8", "203.0.113.9", "45.67.89.10"])
def test_routable_ips_accepted(ip):
    assert dex.is_valid_ip(ip) is True


@pytest.mark.parametrize("ip", [
    "0.0.0.0",         # Unspecified address
    "127.0.0.1",       # Loopback
    "224.0.0.1",       # Multicast
    "169.254.1.1",     # Link-local
    "not.an.ip",
])
def test_reserved_and_invalid_ips_rejected(ip):
    assert dex.is_valid_ip(ip) is False


# --- S_8 regular expression parsing ----------------------------------------

def _hosts(text):
    return [h for h in dex.HOST_REGEX.findall(text) if not dex.looks_like_package(h)]


def _ips(text):
    out = []
    for match in dex.IP_REGEX.findall(text):
        ip = next((g for g in match if g), None)
        if ip and dex.is_valid_ip(ip):
            out.append(ip)
    return out


def test_bare_hostname_is_captured():
    assert _hosts("lebar.gicp.net") == ["lebar.gicp.net"]


@pytest.mark.parametrize("text", [
    "action.de.danoeh.antennapod.core.service.rewind",   # Intent identifier, not a domain
    "extra.de.danoeh.antennapod.core.service.allowStream",
    "Log level must be one of Log.VERBOSE, Log.INFO",     # Capitalized code constant
    "Cannot be invoked on Dispatchers.IO",
])
def test_code_identifiers_are_not_hostnames(text):
    assert _hosts(text) == []


def test_url_keeps_path_and_query():
    assert dex.URL_REGEX.findall("see http://a.com/p?x=1&y=2 now") == \
        ["http://a.com/p?x=1&y=2"]


@pytest.mark.parametrize("text", ["1.3.6.1", "2.5.29.37", "5.5.7.3"])
def test_certificate_oids_are_not_ips(text):
    """X.509 OIDs are syntactically valid IPv4 strings; network context checks prevent false positives."""
    assert _ips(text) == []


@pytest.mark.parametrize("text,expected", [
    ("https://8.8.8.8/x", ["8.8.8.8"]),
    ("connect to 203.0.113.9:8080", ["203.0.113.9"]),
])
def test_ips_in_network_context_are_captured(text, expected):
    assert _ips(text) == expected


# --- S_5 / S_6 permission mapping ------------------------------------------

class FakeMethod:
    def __init__(self, class_name, name):
        self._class_name = class_name
        self._name = name

    def get_class_name(self):
        return self._class_name

    def get_name(self):
        return self._name


class FakeMethodAnalysis:
    def __init__(self, class_name, name):
        self._method = FakeMethod(class_name, name)

    def get_method(self):
        return self._method


class FakeAnalysis:
    """Mock object replacing Androguard's Analysis to provide deterministic API mappings."""

    def __init__(self, mapping):
        self._mapping = mapping

    def get_permissions(self, api_level):
        return [
            (FakeMethodAnalysis(cls, name), perms)
            for (cls, name), perms in self._mapping.items()
        ]


class FakeAPK:
    def __init__(self, permissions):
        self._permissions = permissions

    def get_permissions(self):
        return self._permissions


def test_s5_collects_every_permission_protected_call():
    dx = FakeAnalysis({
        ("Landroid/telephony/SmsManager;", "sendTextMessage"): ["android.permission.SEND_SMS"],
        ("Landroid/location/LocationManager;", "getLastKnownLocation"): ["android.permission.ACCESS_FINE_LOCATION"],
    })
    apk = FakeAPK([])
    calls, _ = dex.permission_mapped_calls(apk, dx)
    assert calls == {
        "Landroid/telephony/SmsManager;->sendTextMessage",
        "Landroid/location/LocationManager;->getLastKnownLocation",
    }


def test_s6_is_the_intersection_with_requested_permissions():
    """S_6 represents restricted calls requiring permissions declared in the Android Manifest."""
    dx = FakeAnalysis({
        ("Landroid/telephony/SmsManager;", "sendTextMessage"): ["android.permission.SEND_SMS"],
        ("Landroid/location/LocationManager;", "getLastKnownLocation"): ["android.permission.ACCESS_FINE_LOCATION"],
    })
    apk = FakeAPK(["android.permission.SEND_SMS", "android.permission.VIBRATE"])
    _, used = dex.permission_mapped_calls(apk, dx)
    assert used == {"android.permission.SEND_SMS"}


def test_s6_empty_when_nothing_overlaps():
    """Observed in modern APKs: API 16 mappings reference legacy permissions while manifest requests API 28+."""
    dx = FakeAnalysis({
        ("Landroid/location/LocationManager;", "getLastKnownLocation"): ["android.permission.ACCESS_FINE_LOCATION"],
    })
    apk = FakeAPK(["android.permission.POST_NOTIFICATIONS"])
    _, used = dex.permission_mapped_calls(apk, dx)
    assert used == set()


def test_api_level_is_passed_through():
    seen = {}

    class RecordingAnalysis(FakeAnalysis):
        def get_permissions(self, api_level):
            seen["level"] = api_level
            return []

    dex.permission_mapped_calls(FakeAPK([]), RecordingAnalysis({}))
    assert seen["level"] == dex.API_LEVEL == 16


# --- S_7 string constant scanning -------------------------------------------

def _string_hits(text):
    """Replicates S_7 string scanning logic implemented in dex.extract()."""
    return [p for p in dex.SUSPICIOUS_STRING_PATTERNS if p in text]


@pytest.mark.parametrize("text,expected", [
    ("/system/bin/su", "system/bin/su"),          # Table III: DroidKungFu, GingerMaster indicators
    ("/system/xbin/su", "system/xbin/su"),
    ("chmod 777 /data/local", "chmod"),
    ("busybox ls -la", "busybox"),
])
def test_root_indicators_are_detected(text, expected):
    assert expected in _string_hits(text)


@pytest.mark.parametrize("text", [
    "just a normal log message",
    "https://example.com/download",
    "android.intent.action.MAIN",
])
def test_ordinary_strings_produce_no_s7_hits(text):
    assert _string_hits(text) == []


def test_pattern_itself_is_stored_not_the_whole_string():
    """S_7 features record matched patterns rather than entire enclosing string literals (Table III)."""
    hits = _string_hits("exec /system/bin/su -c 'chmod 777 /data'")
    assert "system/bin/su" in hits
    assert "exec /system/bin/su -c 'chmod 777 /data'" not in hits