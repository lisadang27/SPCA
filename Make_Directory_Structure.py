#!/usr/bin/env python
# coding: utf-8

from pathlib import Path

from SPCA.directory_structure import make_directory_structure

basepath = '/homes/picaro/bellt/research/'

planets = ['CoRoT-2b', 'HAT-P-7b', 'KELT-1b', 'KELT-16b', 'KELT-9b', 'MASCARA-1b', 'Qatar-1b', 'WASP-103b', 'WASP-12b', 'WASP-12b_old', 'WASP-14b', 'WASP-18b', 'WASP-19b', 'WASP-33b', 'WASP-43b', 'HD189733b', 'HD209458b', 'HD149026b']

for planet in planets:
    rootpath = Path(basepath) / planet
    print('Starting', rootpath)
    aorSnip = make_directory_structure(rootpath)
    print('Your AOR snippet is', aorSnip)

