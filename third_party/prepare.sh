#!/bin/bash -e

cd $(dirname $0)

# force use /usr/bin/sort on windows, /c/Windows/system32/sort do not support -V
OS=$(uname -s)
SORT=sort
if [[ $OS =~ "NT" ]]; then
    SORT=/usr/bin/sort
fi

requiredGitVersion="1.8.4"
currentGitVersion="$(git --version | awk '{print $3}')"
if [ "$(printf '%s\n' "$requiredGitVersion" "$currentGitVersion" | ${SORT} -V | head -n1)" = "$currentGitVersion" ]; then
    echo "Please update your Git version. (foud version $currentGitVersion, required version >= $requiredGitVersion)"
    exit -1
fi

# BEGIN-INTERNAL
echo "Start downloading git submodules"
# END-INTERNAL
function git_submodule_update() {
    git submodule sync
    git submodule update -f --init MegEngine
    git submodule update -f --init progressbar
}

git_submodule_update

bash -c "sed -i \"s/if(MGE_INFERENCE_ONLY AND NOT MGE_WITH_TEST)/if(MGE_WITH_MINIMUM_SIZE)/g\" MegEngine/CMakeLists.txt"

# BEGIN-INTERNAL
echo "Finished downloading git submodules"
# END-INTERNAL
