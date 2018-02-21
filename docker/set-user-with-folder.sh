#!/usr/bin/env sh

#WHO=/docker_user
#WHO="$HOST_CORPORA_OUT"
#WHO=/root/corpora_out
WHO=$1
shift

stat $WHO > /dev/null || (echo You must mount a file to "$WHO" in order to properly assume user && exit 1)
echo "out_path=$WHO"

echo "args:"
echo "$@"

USERID=$(stat -c %u $WHO)
GROUPID=$(stat -c %g $WHO)

echo "userid=$USERID"
echo "groupid=$GROUPID"


#groupadd -g $GROUPID docker_user
#id -u somegroupname &>/dev/null
GROUPIDROOT=$(id -g root)
echo "GROUPIDROOT=$GROUPIDROOT"
if [ "$GROUPIDROOT" -ne "$GROUPID" ]; then
    deluser docker_user > /dev/null 2>&1
    echo "deleted docker_user"
    addgroup --gid $GROUPID docker_user
    echo "added group docker_user"
    #useradd -u $USERID -G docker_user -D -s /bin/sh docker_user
    useradd --uid $USERID --gid $GROUPID docker_user
    echo "added user docker_user"
    gosu docker_user "$@"
else
    #echo "could not add group docker_user with groupid=$GROUPID, assume to be root"
    echo "user group is root, execute as root"
    gosu root "$@"
fi

