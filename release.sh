#!/bin/bash
version_old=$(git describe --tags --abbrev=0)
# if [[ $(git status -s) ]]; then
#     echo "Error: Git working directory is not clean. Please commit or discard your changes."
#     exit 1
# fi
bumpversion $1 --allow-dirty
version_new=v$(dotnet-gitversion /showvariable MajorMinorPatch)
git add .
git commit -n -m "Bumpversion $version_old -> $version_new"
git push
git tag -a $version_new -m "Release $version_new"
git push origin $version_new