#!/bin/bash
# usage: run_grader_job.sh <tag> <port> <workers> <cmd...>   (server + client on the same node; token never printed)
TAG=$1; PORT=$2; W=$3; shift 3
PY=/it-share-prj2-1/data/goramesh-instella-rl/out/v7_grader/venv/bin/python
SRV=/it-share-prj2-1/data/goramesh-instella-rl/out/v7_regrade/code/code_exec_server.py
S=/tmp/v7audit_$TAG; rm -rf $S; mkdir -p $S; chmod 700 $S
python3 $SRV --host 127.0.0.1 --port $PORT --workers $W --max-tests 64 --max-timeout 20 --python $PY \
  --token-file $S/token --scratch $S/scratch > /it-share-prj2-1/data/goramesh-instella-rl/out/v7_improve/data_audit/server_$TAG.log 2>&1 &
SP=$!; trap 'kill $SP; sleep 2; rm -rf $S' EXIT
for i in $(seq 100); do [ -s $S/token ] && break; sleep 0.2; done; sleep 2
export V7_GRADER_URL="http://127.0.0.1:$PORT/$(tr -d '[:space:]' < $S/token)"
"$@"
