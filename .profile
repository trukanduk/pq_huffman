if [ "$_PQ_PROFILE" ]
then
    return
fi

_YEAL_PATH=/home/ilya/yael/yael_v438/yael
if [ -z "$PYTHONPATH" ]
then
    export PYTHONPATH='.'
fi

export PYTHONPATH="$PYTHONPATH:$_YEAL_PATH"

export _PQ_PROFILE=1
export PS1="(pq) $PS1"
