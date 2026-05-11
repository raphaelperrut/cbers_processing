# -*- coding: utf-8 -*-

def classFactory(iface):
    from .cbers_colorize_plugin import CBERSColorizePlugin
    return CBERSColorizePlugin(iface)