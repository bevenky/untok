"""Regenerate untok's offline script policy from pinned Unicode 17 data.

Usage: python scripts/generate_bundle_script_ranges.py UCD_DIRECTORY OUTPUT_PY
The input directory must contain the four exact files named in SOURCES.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

SOURCES = {
    "UnicodeData.txt": "2e1efc1dcb59c575eedf5ccae60f95229f706ee6d031835247d843c11d96470c",
    "Scripts.txt": "9f5e50d3abaee7d6ce09480f325c706f485ae3240912527e651954d2d6b035bf",
    "ScriptExtensions.txt": "ec2107e58825a1586acee8e0911ce18260394ac8b87e535ca325f1ccbeb06bc6",
    "PropertyValueAliases.txt": "64e9a5f76f7a1e8b5a47d6a1f9a26522a251208f5276bdfa1559dac7cf2e827a",
}
UNICODE_LICENSE = 'UNICODE LICENSE V3\n\nCOPYRIGHT AND PERMISSION NOTICE\n\nCopyright © 1991-2026 Unicode, Inc.\n\nNOTICE TO USER: Carefully read the following legal agreement. BY\nDOWNLOADING, INSTALLING, COPYING OR OTHERWISE USING DATA FILES, AND/OR\nSOFTWARE, YOU UNEQUIVOCALLY ACCEPT, AND AGREE TO BE BOUND BY, ALL OF THE\nTERMS AND CONDITIONS OF THIS AGREEMENT. IF YOU DO NOT AGREE, DO NOT\nDOWNLOAD, INSTALL, COPY, DISTRIBUTE OR USE THE DATA FILES OR SOFTWARE.\n\nPermission is hereby granted, free of charge, to any person obtaining a\ncopy of data files and any associated documentation (the "Data Files") or\nsoftware and any associated documentation (the "Software") to deal in the\nData Files or Software without restriction, including without limitation\nthe rights to use, copy, modify, merge, publish, distribute, and/or sell\ncopies of the Data Files or Software, and to permit persons to whom the\nData Files or Software are furnished to do so, provided that either (a)\nthis copyright and permission notice appear with all copies of the Data\nFiles or Software, or (b) this copyright and permission notice appear in\nassociated Documentation.\n\nTHE DATA FILES AND SOFTWARE ARE PROVIDED "AS IS", WITHOUT WARRANTY OF ANY\nKIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF\nMERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT OF\nTHIRD PARTY RIGHTS.\n\nIN NO EVENT SHALL THE COPYRIGHT HOLDER OR HOLDERS INCLUDED IN THIS NOTICE\nBE LIABLE FOR ANY CLAIM, OR ANY SPECIAL INDIRECT OR CONSEQUENTIAL DAMAGES,\nOR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,\nWHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION,\nARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THE DATA\nFILES OR SOFTWARE.\n\nExcept as contained in this notice, the name of a copyright holder shall\nnot be used in advertising or otherwise to promote the sale, use or other\ndealings in these Data Files or Software without prior written\nauthorization of the copyright holder.\n'

INDIC_SCRIPTS = ("Arabic", "Bengali", "Devanagari", "Gujarati", "Gurmukhi",
                 "Kannada", "Malayalam", "Meetei_Mayek", "Ol_Chiki", "Oriya", "Tamil", "Telugu")


def generate(source: Path, output: Path) -> None:
    texts = {}
    for name, digest in SOURCES.items():
        data = (source / name).read_bytes()
        if hashlib.sha256(data).hexdigest() != digest:
            raise ValueError(f"Wrong pinned Unicode input: {name}")
        texts[name] = data.decode("utf-8")
    aliases = {}
    for line in texts["PropertyValueAliases.txt"].splitlines():
        fields = [x.strip() for x in line.split("#")[0].split(";")]
        if fields[0] == "sc":
            aliases[fields[1]] = fields[2]
    def parse(text):
        for line in text.splitlines():
            line = line.split("#")[0].strip()
            if line:
                span, value = [x.strip() for x in line.split(";")]
                a, *b = span.split("..")
                yield int(a, 16), int(b[0], 16) if b else int(a, 16), value
    scripts, extensions = {}, {}
    for lo, hi, name in parse(texts["Scripts.txt"]):
        scripts.update(dict.fromkeys(range(lo, hi + 1), name))
    for lo, hi, names in parse(texts["ScriptExtensions.txt"]):
        extensions.update(dict.fromkeys(range(lo, hi + 1), {aliases[n] for n in names.split()}))
    punctuation = set()
    for line in texts["UnicodeData.txt"].splitlines():
        fields = line.split(";")
        if fields[2].startswith("P"):
            punctuation.add(int(fields[0], 16))
    def ranges(allowed):
        result = []
        for cp in range(0x110000):
            script = scripts.get(cp, "Unknown")
            keep = script in allowed or (script == "Common" and cp in punctuation) or (script in {"Common", "Inherited"} and
                    (cp not in extensions or bool(extensions[cp] & allowed)))
            if keep:
                if result and result[-1][1] == cp - 1:
                    result[-1][1] = cp
                else:
                    result.append([cp, cp])
        return result
    table = {"latin": ranges({"Latin"}), "latin-indic": ranges({"Latin", *INDIC_SCRIPTS})}
    canonical = json.dumps(table, sort_keys=True, separators=(",", ":")).encode()
    lines = ['"""Generated Unicode 17 script policy. Regenerate with scripts/generate_bundle_script_ranges.py.',
             '', 'Derived from Unicode Character Database files, copyright 2025 Unicode, Inc.',
             'License: https://www.unicode.org/license.txt',
             'Sources: https://www.unicode.org/Public/17.0.0/ucd/', '', *UNICODE_LICENSE.splitlines(), '"""', '',
             'UNICODE_VERSION = "17.0.0"', 'SOURCES = ' + repr(SOURCES),
             'INDIC_SCRIPTS = ' + repr(INDIC_SCRIPTS),
             'TABLE_SHA256 = ' + repr(hashlib.sha256(canonical).hexdigest()), 'RANGES = {']
    for profile, values in table.items():
        lines.append(f'    {profile!r}: (')
        lines.extend(f'        (0x{lo:X}, 0x{hi:X}),' for lo, hi in values)
        lines.append('    ),')
    lines.extend(['}', ''])
    output.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    generate(Path(sys.argv[1]), Path(sys.argv[2]))
