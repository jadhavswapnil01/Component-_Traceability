# Set the PYTHONUTF8 environment variable to 1 to force UTF-8 mode
$env:PYTHONUTF8 = "1"

# Set the JAVA_HOME for this terminal session only
$env:JAVA_HOME = "C:\Java\jdk-21"

# Add the new Java version to the start of the PATH for this session
$env:Path = "$env:JAVA_HOME\bin;" + $env:Path

Write-Host "✅ Java 21 is now active for this terminal."
Write-Host "   JAVA_HOME: $($env:JAVA_HOME)"
Write-Host "   Python UTF-8 Mode: Enabled"

# Activate your Python virtual environment using the PowerShell script
.\venv\Scripts\Activate.ps1

Write-Host "🚀 Your development environment is ready."