if [ "$_PQ_PROFILE" ]
then
    return
fi

if [ "$1" = 'z' ]
then
    export YAEL_HOME=/home/trukanduk/pq/yael/yael_v438
else
    export YAEL_HOME=/home/ilya/yael/yael_v438
fi

if [ "$LD_LIBRARY_PATH" ]
then
    export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$YAEL_HOME/yael"
else
    export LD_LIBRARY_PATH="$YAEL_HOME/yael"
fi

if [ -z "$PYTHONPATH" ]
then
    export PYTHONPATH='.'
fi

export PYTHONPATH="$PYTHONPATH:$YAEL_HOME/yael"

export _PQ_PROFILE=1
export PS1="(pq) $PS1"
