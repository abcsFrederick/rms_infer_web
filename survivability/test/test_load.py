import pytest

from girder.plugin import loadedPlugins


@pytest.mark.plugin('survivability')
def test_import(server):
    assert 'survivability' in loadedPlugins()
