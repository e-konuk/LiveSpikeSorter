# Installs the parts of CUDA 11.8 needed to build LiveSpikeSorter.exe, plus cuDNN 8.9.7,
# on a GitHub Actions Windows runner. Writes CUDA_PATH and CUDNN_PATH to $env:GITHUB_ENV.
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"   # Invoke-WebRequest is very slow with the progress bar

$cudaUrl  = "https://developer.download.nvidia.com/compute/cuda/11.8.0/network_installers/cuda_11.8.0_windows_network.exe"
$cudnnUrl = "https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/windows-x86_64/cudnn-windows-x86_64-8.9.7.29_cuda11-archive.zip"
$cudaPath = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8"
$work     = Join-Path $env:RUNNER_TEMP "cuda"
New-Item -ItemType Directory -Force $work | Out-Null

# CUDA: compiler, the libraries the exe links, the headers it includes, and VS integration
Invoke-WebRequest $cudaUrl -OutFile "$work\cuda.exe"
$components = "nvcc", "cudart", "cublas", "cublas_dev", "cufft", "cufft_dev", "cusolver", "cusolver_dev",
              "cusparse_dev", "npp", "npp_dev", "thrust", "visual_studio_integration" | ForEach-Object { "${_}_11.8" }
Start-Process -Wait -FilePath "$work\cuda.exe" -ArgumentList (@("-s") + $components)
if (-not (Test-Path "$cudaPath\bin\nvcc.exe")) { throw "CUDA install failed: nvcc.exe not found" }

# The installer does not reliably register the CUDA build customizations with VS on runners
$vs = & "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe" -latest -products * -property installationPath
Copy-Item "$cudaPath\extras\visual_studio_integration\MSBuildExtensions\*" `
          "$vs\MSBuild\Microsoft\VC\v170\BuildCustomizations" -Force

# cuDNN: NVIDIA's public redistributable archive (no login needed)
Invoke-WebRequest $cudnnUrl -OutFile "$work\cudnn.zip"
Expand-Archive "$work\cudnn.zip" -DestinationPath $work -Force
$cudnnPath = Get-ChildItem $work -Directory -Filter "cudnn-windows-*" | Select-Object -First 1 -ExpandProperty FullName
if (-not (Test-Path "$cudnnPath\include\cudnn.h")) { throw "cuDNN archive layout not recognized" }

"CUDA_PATH=$cudaPath"   | Out-File -Append -Encoding utf8 $env:GITHUB_ENV
"CUDNN_PATH=$cudnnPath" | Out-File -Append -Encoding utf8 $env:GITHUB_ENV
