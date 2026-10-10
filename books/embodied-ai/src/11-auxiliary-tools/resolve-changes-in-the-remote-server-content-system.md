# Изменился ключ удалённого сервера

Если сервер переустановлен или изменились его ключи хоста, SSH перестаёт подключаться и выводит предупреждение. Ниже пример такого вывода (адрес и путь заменены на плейсхолдеры):

```
2025-09-25 16:14:08.992 [info] (ssh_tunnel) stdout: @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
@    WARNING: REMOTE HOST IDENTIFICATION HAS CHANGED!     @
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
IT IS POSSIBLE THAT SOMEONE IS DOING SOMETHING NASTY!
Someone could be eavesdropping on you right now (man-in-the-middle attack)!
It is also possible that a host key has just been changed.
The fingerprint for the ED25519 key sent by the remote host is
SHA256:<fingerprint>.
Please contact your system administrator.
Add correct host key in C:\\Users\\<user>/.ssh/known_hosts to get rid of this message.  
Offending ECDSA key in C:\\Users\\<user>/.ssh/known_hosts:134
Host key for [<host>]:<port> has changed and you have requested strict checking. [14;1H
```

Прежде чем что-либо менять, убедитесь, что сервер действительно переустановлен или его ключи были обновлены администратором. Иначе это может быть атака «человек посередине».

Если это ожидаемое изменение, удалите старый ключ из `known_hosts` и подключитесь заново:

```bash
ssh-keygen -R "[<host>]:<port>"

ssh <host> -p <port>
```
