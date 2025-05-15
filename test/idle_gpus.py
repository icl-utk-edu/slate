#!/usr/bin/env python3
#
# Prints the GPU kind (cuda or rocm), then space-delimited ids of
# idle GPUs, i.e., that do not have a process.
# Used in CI test.sh and gpu_bind.sh scripts.

import subprocess
import re
import shutil

ngpus = 0
gpu_kind = 'unknown'

# Search for CUDA GPUs.
if (shutil.which( 'nvidia-smi' )):
    p = subprocess.run( ['nvidia-smi', '--list-gpus'], capture_output=True )
    stdout = p.stdout.decode('utf-8').splitlines()
    ngpus = len( list( filter( lambda line: re.search( 'GPU', line ), stdout ) ) )
    if (ngpus > 0):
        gpu_kind = 'cuda'
# end

# If no CUDA GPUs, search for ROCm GPUs.
if (ngpus == 0 and shutil.which( 'rocm-smi' )):
    p = subprocess.run( ['rocm-smi', '--showuniqueid'], capture_output=True )
    stdout = p.stdout.decode('utf-8').splitlines()
    seen = {}
    ngpus = 0
    for line in stdout:
        s = re.search( r'(GPU\[\d+\])', line )
        if (s and s.group(1) not in seen):
            seen[ s.group(1) ] = True
            ngpus += 1
    if (ngpus > 0):
        gpu_kind = 'rocm'
# end

# Initially, mark all GPUs as idle.
gpus = { str( gpu ): 1 for gpu in range( ngpus ) }

if (gpu_kind == 'cuda'):
    # Mark GPUs that have processes.
    p = subprocess.run( 'nvidia-smi', capture_output=True )
    stdout = p.stdout.decode('utf-8').splitlines()
    section = 1
    for line in stdout:
        if (re.search( '^\| Processes:', line )):
            section = 2
        if (section == 1):
            # Match GPU lines:
            # |   0  Tesla V100-SXM2-32GB            On | 00000000:06:00.0 Off |                    0 |
            # | N/A   39C    P0               60W / 300W|  30487MiB / 32768MiB |      0%      Default |
            s = re.search( '^\| +(\d+) ', line )
            if (s):
                gpu = s.group( 1 )

            # If using >= 50% memory or utilization, mark it as not idle.
            # This allows some sharing of GPUs with other users.
            # Typically idle is 1 MiB and 0% utilization.
            # Docker can't see processes in section 2.
            s = re.search( '^\| +N/A +\d+C +\w+ +\d+W +/ +\d+W *\| +(\d+)MiB +/ +(\d+)MiB *\| +(\d+)%', line )
            if (s):
                used_mem  = int( s.group( 1 ) )
                total_mem = int( s.group( 2 ) )
                percent   = int( s.group( 3 ) )
                if (used_mem >= 0.5*total_mem or percent >= 50):
                    gpus[ gpu ] = 0
        else:
            # Match process lines:
            # |    0   N/A  N/A   1768154      C   application    30482MiB |
            s = re.search( '^\| +(\d) ', line )
            if (s):
                gpus[ s.group( 1 ) ] = 0

elif (gpu_kind == 'rocm'):
    # Mark GPUs that have processes.
    p = subprocess.run( ['rocm-smi', '--showpidgpus'], capture_output=True )
    stdout = p.stdout.decode('utf-8').splitlines()
    process = False
    for line in stdout:
        # Match process lines:
        # PID 2167109 is using 2 DRM device(s):
        # 1 0
        if (re.search( '^PID \d+ is using', line )):
            process = True
        if (process):
            for gpu in re.findall( r'(\d+)', line ):
                gpus[ gpu ] = 0
            process = False
# end

idle_gpus = filter( lambda key: gpus[ key ], gpus )
print( gpu_kind, ' '.join( idle_gpus ) )
