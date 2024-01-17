"""
This modules enables interactive debugging on failing tests.
It is tailored at the workflow of the main developer.
Delete it if it causes problems.

It is only active if a special environment variable is "True"
"""

import os
if os.getenv("PYTEST_IPS") == "True":

    import ipydex

    def pytest_runtest_setup(item):
        print("This invocation of pytest is customized")


    def pytest_exception_interact(node, call, report):
        ipydex.ips_excepthook(call.excinfo.type, call.excinfo.value, call.excinfo.tb, frame_upcount=0)
