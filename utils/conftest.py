import pytest


def pytest_addoption(parser):
    parser.addoption("--slow", action="store_true", default=False, help="run slow tests")
    parser.addoption("--opt", action="store_true", default=False, help="run opt tests")
    parser.addoption("--old", action="store_true", default=False, help="run old tests")
    parser.addoption(
        "--lvl", default=0, help=" test levels"
    )
    parser.addoption("--new", action="store_true", default=False, help="run only new tests")


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow to run")
    config.addinivalue_line("markers", "opt: mark test as optional")
    config.addinivalue_line("markers", "old: mark test as old")
    for i in range(1,3):
        config.addinivalue_line("markers", "lvl%d: test level %d" %(i,i))
    config.addinivalue_line("markers", "new: mark test as new to run")

def pytest_collection_modifyitems(config, items):
    if config.getoption("--new") :
        skip_all = pytest.mark.skip(reason="not a --new test")
        for item in items:
            if "new" in item.keywords:
                print('will run')
            else:
                item.add_marker(skip_all)
        return

    if not config.getoption("--slow") :
        # --runslow given in cli: do not skip slow tests
        skip_slow = pytest.mark.skip(reason="need --runslow option to run")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)

    if not config.getoption("--old") :
        # --runslow given in cli: do not skip slow tests
        skip_slow = pytest.mark.skip(reason="need --old to run")
        for item in items:
            if "old" in item.keywords:
                item.add_marker(skip_slow)

    if not config.getoption("--opt") :
        skip_opt = pytest.mark.skip(reason="optional test")
        for item in items:
            if "opt" in item.keywords:
                item.add_marker(skip_opt)

    lvl = int(config.getoption("--lvl"))
    for item in items:
        for i in range(1,3):
            if "lvl%d" %i in item.keywords:
                item.add_marker(pytest.mark.skipif(lvl<i, reason="test level %d<%d" %(lvl,i)))
                break
