#!/bin/bash
version_old=$(git describe --tags $(git rev-list --tags --max-count=1))
bumpversion $1
version_new=v$(dotnet-gitversion /showvariable MajorMinorPatch)
git add .
git commit -n -m "Bumpversion $version_old -> $version_new"
git push
git tag -a $version_new -m "Release $version_new"
git push origin $version_new