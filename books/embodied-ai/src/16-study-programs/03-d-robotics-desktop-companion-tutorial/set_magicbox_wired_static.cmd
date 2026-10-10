@echo off
netsh interface ip set address name="以太网" static 192.168.127.100 255.255.255.0 none
netsh interface ip set dns name="以太网" dhcp
echo.
echo IP-адрес проводного подключения Magicbox настроен на адаптере "以太网" [Ethernet].
echo Test with: ping 192.168.127.10
pause
