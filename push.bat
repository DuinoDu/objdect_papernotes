@echo off
set /p msg="Enter commit msg: "
git add .
git commit -m "%msg%"
git push