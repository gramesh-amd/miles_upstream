#!/usr/bin/env python3
"""List one level of a bucket prefix using a DELIMITER, not a full enumeration.

WHY THIS EXISTS. oci_put.py --count pages through every object under a prefix
to sum sizes. Under goramesh/ that is 280k+ objects (.local alone has 188k,
.triton 84k), so a one-level question costs minutes. list_objects_v2 with
Delimiter="/" returns CommonPrefixes -- the folder names -- in one or two
round trips regardless of how much is underneath.

Use --count only when you actually need sizes for a known, small prefix.

    python3 oci_ls.py creds.env goramesh/
    python3 oci_ls.py creds.env goramesh/instella-rl-mi325x/

Also fetches ONE object, because oci_put.py --get takes a PREFIX and appends
"/" to it, so passing a full object key there matches nothing and reports
"0 objects" -- which looks like a missing file rather than a misuse:

    python3 oci_ls.py creds.env --get <object-key> <local-path> [--head N]

--head N downloads only the first N bytes, which is how you inspect the schema
of a 2 GB generations file without pulling 2 GB.

The key is read from a file and never passed on argv or printed.
"""
import os
import sys

import boto3
from botocore.config import Config


def load(path):
    env = {}
    for line in open(path):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            env[k] = v
    return env


env = load(sys.argv[1])
prefix = sys.argv[2] if len(sys.argv) > 2 else ""
s3 = boto3.client(
    "s3",
    endpoint_url=env["EP"],
    aws_access_key_id=env["AK1"],
    aws_secret_access_key=env["SK1"],
    config=Config(signature_version="s3v4",
                  request_checksum_calculation="when_required",
                  retries={"max_attempts": 5, "mode": "standard"}),
)
bucket = env.get("BUCKET") or env.get("B1")

if prefix == "--get":
    key, dest = sys.argv[3], sys.argv[4]
    nbytes = None
    if "--head" in sys.argv:
        nbytes = int(sys.argv[sys.argv.index("--head") + 1])
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    if nbytes:
        # Range request: inspect the head of a huge file without fetching it all.
        body = s3.get_object(Bucket=bucket, Key=key,
                             Range=f"bytes=0-{nbytes - 1}")["Body"].read()
        open(dest, "wb").write(body)
        print(f"  wrote {len(body)} bytes (head) -> {dest}")
    else:
        s3.download_file(bucket, key, dest)
        print(f"  wrote {os.path.getsize(dest)} bytes -> {dest}")
    sys.exit(0)

tok = None
dirs, files = [], []
while True:
    kw = dict(Bucket=bucket, Prefix=prefix, Delimiter="/", MaxKeys=1000)
    if tok:
        kw["ContinuationToken"] = tok
    r = s3.list_objects_v2(**kw)
    dirs += [p["Prefix"] for p in r.get("CommonPrefixes", [])]
    files += [(o["Key"], o["Size"]) for o in r.get("Contents", [])]
    if not r.get("IsTruncated"):
        break
    tok = r["NextContinuationToken"]

print(f"prefix: {prefix!r}   ({len(dirs)} folders, {len(files)} files at this level)")
for d in sorted(dirs):
    print(f"  DIR   {d}")
for k, sz in sorted(files)[:60]:
    print(f"  FILE  {k}   {sz / 2**20:.1f} MB")
