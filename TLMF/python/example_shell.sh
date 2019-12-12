#!/bin/bash
#打印菜单
while :
do
 python creep_content.py


 echo "********************"
 echo "    menu    "
 echo "1.tima and date"
 echo "2.system info"
 echo "3.uesrs are doing"
 echo "4.exit"
 echo "********************"
 read -p "enter you choice [1-4]:" choice
#根据客户的选择做相应的操作
 case $choice in
  1)
  echo "today is `date +%Y-%m-%d`"
  echo "time is `date +%H:%M:%S`"
  read -p "press [enter] key to continue..." Key  #暂停循环，提示客户按enter键继续
  ;;
  2)
  uname -r
  read -p "press [enter] key to continue..." Key
  ;;
  3)
  w
  read -p "press [enter] key to continue..." Key
  ;;
  4)
  echo "Bye!"
  exit 0
  ;;
  *)
  echo "error"
  read -p "press [enter] key to continue..." Key
  ;;
 esac
done