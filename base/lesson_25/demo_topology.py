import argparse
import csv
import re
import subprocess


QUERY_FIELDS = "index,name,driver_version,memory.total,pcie.link.gen.current,pcie.link.width.current"
EXAMPLE = """\
        GPU0    GPU1    CPU Affinity    NUMA Affinity   GPU NUMA ID
GPU0    X       SYS     0-15            0               N/A
GPU1    SYS     X       16-31           1               N/A
"""


def parse_gpu_rows(text):
    clean = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", text)
    lines = clean.splitlines()
    for offset, line in enumerate(lines):
        if "CPU Affinity" in line and "NUMA Affinity" in line:
            peers = line.split("CPU Affinity", 1)[0].split()
            break
    else:
        raise ValueError("missing CPU/NUMA affinity headers in nvidia-smi output")
    gpus = [name for name in peers if re.fullmatch(r"GPU\d+", name)]
    if not gpus:
        raise ValueError("no GPU columns in the topology matrix")

    rows = {}
    for line in lines[offset + 1:]:
        fields = line.split()
        if not fields or fields[0] not in gpus:
            continue
        name, *values = fields
        if name in rows or len(values) < len(peers) + 2:
            raise ValueError(f"invalid topology row for {name}")
        links = dict(zip(peers, values[:len(peers)], strict=True))
        cpu, numa = values[len(peers):len(peers) + 2]
        rows[name] = (links, cpu, numa)
    if rows.keys() != set(gpus):
        raise ValueError("topology GPU rows disagree with the columns")
    return gpus, rows


def print_topology(text):
    gpus, rows = parse_gpu_rows(text)
    for i, name in enumerate(gpus):
        links, cpu, numa = rows[name]
        if links[name] != "X":
            raise ValueError(f"{name} is missing the self link")
        print(f"{name}: CPU affinity={cpu}, NUMA affinity={numa}")
        for peer in gpus[i + 1:]:
            if links[peer] != rows[peer][0][name]:
                raise ValueError(f"asymmetric link labels for {name} and {peer}")
            print(f"{name} -> {peer}: {links[peer]}")
    if len(gpus) == 1:
        print("one visible GPU: no pairwise GPU route")


def nvidia_smi(*arguments):
    return subprocess.check_output(["nvidia-smi", *arguments], text=True, timeout=15)


def main():
    parser = argparse.ArgumentParser(description="Read selected nvidia-smi topology fields")
    parser.add_argument("--cpu-smoke", action="store_true")
    args = parser.parse_args()
    if args.cpu_smoke:
        print("CPU smoke: fictional dual-GPU topology, no hardware query")
        print_topology(EXAMPLE)
        print("topology parser: pass")
        return
    try:
        report = nvidia_smi("--query-gpu=" + QUERY_FIELDS, "--format=csv,noheader")
        topology = nvidia_smi("topo", "-m")
    except FileNotFoundError as error:
        raise SystemExit("nvidia-smi was not found. Use --cpu-smoke for the parser check.") from error
    for line in csv.reader(report.splitlines(), skipinitialspace=True):
        index, name, driver, memory, gen, width = line
        print(f"GPU{index}: {name}, driver={driver}, total={memory}, PCIe gen={gen}, width={width}")
    print_topology(topology)
    print("topology parser: pass")


if __name__ == "__main__":
    main()
