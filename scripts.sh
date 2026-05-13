# train
$logDir = "G:\nblongT04\LPDGAN\logs_train"
New-Item -ItemType Directory -Path $logDir -Force | Out-Null

$p = Start-Process uv -ArgumentList @(
    "run", "main.py"
) -RedirectStandardOutput "$logDir\train.log" `
  -RedirectStandardError "$logDir\train.err" `
  -PassThru

$p.Id | Out-File "$logDir\train.pid"
