import importlib
import sys
import types
import unittest
from unittest.mock import patch


class _DummyMCP:
    def __init__(self, *_args, **_kwargs):
        pass

    def tool(self):
        return lambda function: function

    def run(self):
        pass


fastmcp = types.ModuleType("mcp.server.fastmcp")
fastmcp.FastMCP = _DummyMCP
server = types.ModuleType("mcp.server")
server.fastmcp = fastmcp
mcp = types.ModuleType("mcp")
mcp.server = server
mcp.ClientSession = type("ClientSession", (), {})
mcp.StdioServerParameters = type("StdioServerParameters", (), {})
mcp_types = types.ModuleType("mcp.types")
mcp_types.Tool = type("Tool", (), {})
mcp_client = types.ModuleType("mcp.client")
mcp_stdio = types.ModuleType("mcp.client.stdio")
mcp_stdio.stdio_client = lambda *_args, **_kwargs: None
mcp.client = mcp_client
sys.modules.setdefault("mcp", mcp)
sys.modules.setdefault("mcp.server", server)
sys.modules.setdefault("mcp.server.fastmcp", fastmcp)
sys.modules.setdefault("mcp.types", mcp_types)
sys.modules.setdefault("mcp.client", mcp_client)
sys.modules.setdefault("mcp.client.stdio", mcp_stdio)

daily = importlib.import_module("backend.mcp_daily_tools")


class DailyNetworkSecurityTests(unittest.TestCase):
    def test_private_local_and_secret_urls_are_blocked(self):
        blocked = (
            "http://127.0.0.1/private",
            "http://localhost/private",
            "file:///etc/passwd",
            "https://example.com/?token=secret",
            "https://user:password@example.com/",
        )
        for url in blocked:
            with self.subTest(url=url), self.assertRaises(ValueError):
                daily._validated_url(url)

    def test_redirect_destination_is_revalidated(self):
        class Response:
            status_code = 302
            headers = {"location": "http://127.0.0.1/admin"}

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

        class Client:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def stream(self, *_args, **_kwargs):
                return Response()

        with patch.object(daily.httpx, "Client", return_value=Client()), patch.object(
            daily, "_public_addresses", return_value=["93.184.216.34"]
        ):
            with self.assertRaises(ValueError):
                daily._bounded_get("https://example.com/", ("text/html",))

    def test_page_parser_omits_script_and_style_text(self):
        parser = daily._PageTextParser()
        parser.feed(
            "<html><title>Safe</title><script>steal()</script>"
            "<style>.hidden{}</style><body>Visible text</body></html>"
        )
        parser.close()
        self.assertIn("Visible text", parser.text)
        self.assertNotIn("steal()", parser.text)

    def test_bing_fallback_parser_extracts_results(self):
        parser = daily._SearchResultParser()
        parser.feed(
            '<ol><li class="b_algo"><a class="tilk" href="https://example.com/a">'
            '</a><h2><a href="https://example.com/a">'
            "Example A</a></h2><div class=\"b_caption\"><p>Snippet A</p>"
            "</div></li></ol>"
        )
        parser.close()
        self.assertEqual(parser.results[0]["title"], "Example A")
        self.assertEqual(parser.results[0]["snippet"], "Snippet A")

    def test_network_calls_are_rate_limited(self):
        with patch.object(
            daily, "_NETWORK_CALL_TIMES", [0.0] * daily.MAX_NETWORK_CALLS_PER_MINUTE
        ), patch.object(daily.time, "monotonic", return_value=1.0):
            with self.assertRaises(ValueError):
                daily._allow_network_call()


if __name__ == "__main__":
    unittest.main()
