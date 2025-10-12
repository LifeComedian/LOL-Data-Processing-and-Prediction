import os
import sys
import json
import glob
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np

ELITE_MONSTER_MAP = {
    "DRAGON": "dragon_kills",
    "ELDER_DRAGON": "dragon_kills",
    "BARON_NASHOR": "baron_kills",
    "BARON": "baron_kills",
    "RIFTHERALD": "rift_herald_kills",
    "HERALD": "rift_herald_kills",
    "ATAKHAN": "atakhan_kills",
}

BASE_STATS = [
    'kills',
    'deaths',
    'assists',
    'gold_earned',
    'total_damage_dealt_to_champions',
    'total_damage_taken',
    'vision_score',
    'total_minions_killed',
]

TEAM_PREFIXES = ['blue', 'red']
FEATURE_ORDER = [f"{team}_{stat}" for team in TEAM_PREFIXES for stat in BASE_STATS]
DEFAULT_NUM_IMAGES = 10

def parse_match_timeline(json_path: str) -> Dict[str, Any]:
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    info = data.get('info', data)
    frames = info.get('frames', [])
    metadata = data.get('metadata', {})
    match_id = metadata.get('matchId') or info.get('gameId') or os.path.splitext(os.path.basename(json_path))[0]
    puuids = metadata.get('participants', [])
    return {"match_id": match_id, "frames": frames, "puuids": puuids}

def infer_winning_team(frames: List[Dict[str, Any]]) -> int:
    for frame in frames[::-1]:
        for ev in frame.get('events', []):
            if ev.get('type') == 'GAME_END':
                wt = ev.get('winningTeam')
                if wt in (100, 200):
                    return wt
    return -1

def participant_team(pid: int) -> int:
    return 100 if 1 <= pid <= 5 else 200

def load_champion_map_from_details(match_id: str, details_dir: Optional[str]) -> Dict[int, Dict[str, Any]]:
    result = {pid: {"champion_id": None, "champion_name": None} for pid in range(1, 11)}
    if not details_dir:
        return result
    path = os.path.join(details_dir, f"{match_id}.json")
    if not os.path.isfile(path):
        candidates = sorted(glob.glob(os.path.join(details_dir, f"*{match_id}*.json")))
        if not candidates:
            return result
        path = candidates[0]
    try:
        with open(path, 'r', encoding='utf-8') as f:
            det = json.load(f)
        info = det.get('info', det)
        participants = info.get('participants', [])
        for p in participants:
            pid = p.get('participantId')
            if isinstance(pid, int) and 1 <= pid <= 10:
                result[pid]["champion_id"] = p.get('championId')
                result[pid]["champion_name"] = p.get('championName')
    except Exception:
        pass
    return result

def build_overview_index(overview_csv: Optional[str]) -> Optional[pd.DataFrame]:
    if not overview_csv:
        return None
    try:
        df = pd.read_csv(overview_csv)
    except Exception:
        try:
            df = pd.read_csv(overview_csv, encoding="utf-8-sig")
        except Exception:
            df = pd.read_csv(overview_csv, sep=";")
    needed = ["match_id","participant_puuid","champion_id","champion_name"]
    for c in needed:
        if c not in df.columns:
            raise ValueError(f"overview_csv missing column: {c}")
    df_small = df[needed].dropna(subset=["match_id","participant_puuid"]).copy()
    df_small["match_id"] = df_small["match_id"].astype(str)
    df_small["participant_puuid"] = df_small["participant_puuid"].astype(str)
    return df_small

def build_cumulative_tables(match_id: str, frames: List[Dict[str, Any]], puuids: List[str],
                            details_dir: Optional[str], overview_df: Optional[pd.DataFrame]) -> List[pd.DataFrame]:
    pids = set()
    if frames:
        pf0 = frames[0].get('participantFrames', {})
        for k in pf0.keys():
            try:
                pids.add(int(k))
            except Exception:
                pass
    if not pids:
        pids = set(range(1, 10+1))

    winning_team = infer_winning_team(frames)
    win_map = {pid: (1 if participant_team(pid) == winning_team else 0) if winning_team in (100,200) else -1
               for pid in pids}

    champ_map = load_champion_map_from_details(match_id, details_dir)

    if overview_df is not None and puuids:
        sub = overview_df.loc[overview_df["match_id"] == match_id]
        puuid_to_champ = {row["participant_puuid"]: (row.get("champion_id"), row.get("champion_name"))
                          for _, row in sub.iterrows()}
        for idx, puuid in enumerate(puuids, start=1):
            if puuid in puuid_to_champ:
                cid, cname = puuid_to_champ[puuid]
                champ_map[idx]["champion_id"] = cid
                champ_map[idx]["champion_name"] = cname

    cum = {
        pid: {
            "champion_kills": 0,
            "assists": 0,
            "deaths": 0,
            "turret_destroyed": 0,
            "dragon_kills": 0,
            "baron_kills": 0,
            "atakhan_kills": 0,
        } for pid in pids
    }

    dfs: List[pd.DataFrame] = []

    for minute_idx, frame in enumerate(frames, start=1):
        for ev in frame.get('events', []):
            et = ev.get('type')
            if et == 'CHAMPION_KILL':
                killer = ev.get('killerId')
                if killer in cum:
                    cum[killer]["champion_kills"] += 1
                victim = ev.get('victimId')
                if victim in cum:
                    cum[victim]["deaths"] += 1
                for aid in ev.get('assistingParticipantIds', []) or []:
                    if aid in cum:
                        cum[aid]["assists"] += 1
            elif et == 'BUILDING_KILL':
                if ev.get('buildingType') == 'TOWER_BUILDING':
                    killer = ev.get('killerId')
                    if killer in cum:
                        cum[killer]["turret_destroyed"] += 1
            elif et == 'ELITE_MONSTER_KILL':
                killer = ev.get('killerId')
                mtype = ev.get('monsterType')
                key = ELITE_MONSTER_MAP.get(mtype)
                if killer in cum and key:
                    if key in ("dragon_kills", "baron_kills", "atakhan_kills"):
                        cum[killer][key] += 1

        pf = frame.get('participantFrames', {})
        rows = []
        for pid in sorted(pids):
            f = pf.get(str(pid), {})
            damageStats = f.get('damageStats', {})
            damage_dealt = damageStats.get('totalDamageDoneToChampions')
            if damage_dealt is None:
                damage_dealt = damageStats.get('totalDamageDone', 0)
            total_gold = f.get('totalGold', f.get('currentGold', 0))

            row = {
                "participantId": pid,
                "minute": minute_idx,
                "champion_kills": cum[pid]["champion_kills"],
                "deaths": cum[pid]["deaths"],
                "assists": cum[pid]["assists"],
                "minion_kills": f.get('minionsKilled', 0),
                "jungle_minions_killed": f.get('jungleMinionsKilled', 0),
                "turret_destroyed": cum[pid]["turret_destroyed"],
                "dragon_kills": cum[pid]["dragon_kills"],
                "baron_kills": cum[pid]["baron_kills"],
                "atakhan_kills": cum[pid]["atakhan_kills"],
                "damage_dealt": damage_dealt,
                "total_gold": total_gold,
                "xp": f.get('xp', 0),
                "win": win_map[pid],
                "champion_id": champ_map.get(pid, {}).get("champion_id"),
                "champion_name": champ_map.get(pid, {}).get("champion_name"),
            }
            rows.append(row)
        dfs.append(pd.DataFrame(rows))
    return dfs


def write_per_minute_csvs(match_id: str, per_minute_dfs: List[pd.DataFrame], out_dir: str) -> List[str]:
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for df in per_minute_dfs:
        minute = int(df['minute'].iloc[0])
        path = os.path.join(out_dir, f"{match_id}_min{minute:02d}.csv")
        df.to_csv(path, index=False)
        paths.append(path)
    return paths


def collect_axisless_images(root: Path, num_images: int) -> pd.DataFrame:
    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(f"Axisless root not found: {root}")
    rows: List[Dict[str, Any]] = []
    img_cols = [f"img{i+1}" for i in range(num_images)]
    for match_dir in sorted(root_path.iterdir()):
        if not match_dir.is_dir():
            continue
        participant_paths = sorted(p for p in match_dir.glob('participant_*.png') if p.is_file())
        if len(participant_paths) < num_images:
            continue
        rel_paths = [str(p.relative_to(root_path)) for p in participant_paths[:num_images]]
        row = {'match_id': match_dir.name}
        for idx, rel_path in enumerate(rel_paths, start=1):
            row[f"img{idx}"] = rel_path
        rows.append(row)
    columns = ['match_id', *img_cols]
    return pd.DataFrame(rows, columns=columns) if rows else pd.DataFrame(columns=columns)

def build_feature_table(raw_csv: Path) -> pd.DataFrame:
    raw_path = Path(raw_csv)
    usecols = ['match_id', 'team_side', 'win', *BASE_STATS]
    df = pd.read_csv(raw_path, usecols=usecols)
    df['team_side'] = df['team_side'].str.lower()
    df['win'] = df['win'].astype(str).str.lower() == 'true'
    agg = df.groupby(['match_id', 'team_side'])[BASE_STATS].mean().unstack('team_side')
    agg.columns = [f"{team}_{stat}" for stat, team in agg.columns]
    agg = agg.reset_index()
    missing = [col for col in FEATURE_ORDER if col not in agg.columns]
    if missing:
        raise ValueError(f"Missing expected feature columns: {missing}")
    blue_win = (df[df['team_side'] == 'blue']
                .groupby('match_id')['win']
                .any()
                .astype(int)
                .rename('label'))
    feature_df = agg.merge(blue_win, on='match_id', how='inner')
    order = ['match_id', *FEATURE_ORDER, 'label']
    return feature_df[order]

def split_dataset(df: pd.DataFrame, train_ratio: float, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not 0 < train_ratio < 1:
        raise ValueError('train_ratio must be between 0 and 1')
    if len(df) < 2:
        raise ValueError('Need at least two matches to create a split')
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(df))
    cut = int(round(train_ratio * len(df)))
    cut = max(1, min(len(df) - 1, cut))
    train_df = df.iloc[perm[:cut]].reset_index(drop=True)
    val_df = df.iloc[perm[cut:]].reset_index(drop=True)
    return train_df, val_df

def build_training_dataset(axisless_root: Path,
                            raw_csv: Path,
                            output_dir: Path,
                            num_images: int = DEFAULT_NUM_IMAGES,
                            train_ratio: float = 0.8,
                            seed: int = 42,
                            max_matches: Optional[int] = None) -> Dict[str, Any]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    image_df = collect_axisless_images(axisless_root, num_images)
    if image_df.empty:
        raise ValueError(f"No matches with at least {num_images} images found in {axisless_root}")
    feature_df = build_feature_table(raw_csv)
    dataset_df = image_df.merge(feature_df, on='match_id', how='inner')
    if dataset_df.empty:
        raise ValueError('No overlapping matches between axisless images and raw statistics')
    dataset_df['id'] = dataset_df['match_id']
    img_cols = [f"img{i+1}" for i in range(num_images)]
    ordered_cols = ['id', *img_cols, *FEATURE_ORDER, 'label']
    dataset_df = dataset_df[ordered_cols]
    if max_matches is not None:
        if max_matches < 2:
            raise ValueError('max_matches must be at least 2')
        if len(dataset_df) > max_matches:
            rng = np.random.default_rng(seed)
            take_idx = rng.permutation(len(dataset_df))[:max_matches]
            dataset_df = dataset_df.iloc[take_idx].reset_index(drop=True)
    if len(dataset_df) < 2:
        raise ValueError('Need at least two matches to create train/val splits')
    train_df, val_df = split_dataset(dataset_df, train_ratio, seed)
    train_path = output_path / 'train.csv'
    val_path = output_path / 'val.csv'
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    return {
        'matches_with_images': len(image_df),
        'matches_with_features': len(feature_df),
        'requested_max_matches': max_matches,
        'matches_in_dataset': len(dataset_df),
        'train_rows': len(train_df),
        'val_rows': len(val_df),
        'train_path': str(train_path),
        'val_path': str(val_path),
    }

def build_match_label_map(raw_csv: Path) -> Dict[str, int]:
    df = pd.read_csv(raw_csv, usecols=['match_id', 'team_side', 'win'])
    df['team_side'] = df['team_side'].str.lower()
    df['win'] = df['win'].astype(str).str.lower() == 'true'
    blue = (df[df['team_side'] == 'blue']
            .groupby('match_id')['win']
            .any()
            .astype(int))
    return blue.to_dict()

def ensure_minute_image_set(image_dir: Path, match_id: str, minute_str: str, num_players: int) -> bool:
    if not image_dir.is_dir():
        return False
    expected = [image_dir / f"{match_id}_{minute_str}_{i}.png" for i in range(1, num_players + 1)]
    return all(p.exists() for p in expected)

def build_minute_dataset(timeline_dir: Path,
                          image_root: Path,
                          output_dir: Path,
                          raw_csv: Path,
                          num_players: int = 10,
                          max_matches: Optional[int] = None,
                          details_dir: Optional[Path] = None,
                          overview_csv: Optional[Path] = None) -> Dict[str, Any]:
    timeline_dir = Path(timeline_dir)
    image_root = Path(image_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    overview_df = build_overview_index(str(overview_csv)) if overview_csv else None
    label_map = build_match_label_map(raw_csv)

    matches_written = 0
    minutes_written = 0
    matches_skipped_images = 0

    for timeline_path in sorted(timeline_dir.glob('*.json')):
        match_id = timeline_path.stem
        if max_matches is not None and matches_written >= max_matches:
            break

        match_image_root = image_root / match_id
        if not match_image_root.exists():
            matches_skipped_images += 1
            continue

        parsed = parse_match_timeline(str(timeline_path))
        frames = parsed['frames']
        if not frames:
            continue

        per_minute_dfs = build_cumulative_tables(match_id, frames, parsed['puuids'],
                                                 str(details_dir) if details_dir else None,
                                                 overview_df)
        if not per_minute_dfs:
            continue

        match_output_dir = output_dir / match_id
        written_this_match = 0
        label = label_map.get(match_id, 0)

        for minute_df in per_minute_dfs:
            if minute_df.empty:
                continue
            minute_idx = int(minute_df['minute'].iloc[0])
            minute_str = f"min{minute_idx:02d}"
            minute_image_dir = match_image_root / minute_str
            if not ensure_minute_image_set(minute_image_dir, match_id, minute_str, num_players):
                continue

            minute_df = minute_df.sort_values('participantId').reset_index(drop=True)
            if len(minute_df) != num_players:
                continue

            df_out = minute_df.copy()
            if 'win' in df_out.columns:
                df_out = df_out.drop(columns=['win'])
            champ_name = None
            if 'champion_name' in df_out.columns:
                champ_name = df_out.pop('champion_name')
            df_out['label'] = label
            if champ_name is not None:
                df_out['champion_name'] = champ_name

            columns = ['participantId'] + [c for c in df_out.columns if c not in ['participantId', 'champion_name']]                 + ([ 'champion_name'] if 'champion_name' in df_out.columns else [])
            df_out = df_out[columns]

            csv_path = match_output_dir / f"{match_id}_min{minute_idx:02d}.csv"
            match_output_dir.mkdir(parents=True, exist_ok=True)
            df_out.to_csv(csv_path, index=False)
            written_this_match += 1
            minutes_written += 1

        if written_this_match > 0:
            matches_written += 1
        else:
            if match_output_dir.exists() and not any(match_output_dir.iterdir()):
                match_output_dir.rmdir()

    return {
        'matches_written': matches_written,
        'minutes_written': minutes_written,
        'matches_skipped_images': matches_skipped_images,
        'max_matches': max_matches,
        'output_dir': str(output_dir.resolve()),
    }


def parse_cli_args(argv: List[str]) -> Tuple[argparse.ArgumentParser, argparse.Namespace]:
    parser = argparse.ArgumentParser(description='Data processing utilities for match timelines.')
    subparsers = parser.add_subparsers(dest='command')

    pm_parser = subparsers.add_parser('per-minute', help='Generate per-minute participant CSVs.')
    pm_parser.add_argument('timeline_dir', type=str)
    pm_parser.add_argument('output_root', type=str)
    pm_parser.add_argument('--details-dir', type=str, default=None, help='Directory with match detail JSON files.')
    pm_parser.add_argument('--overview-csv', type=str, default=None, help='CSV with participant metadata overrides.')

    build_parser = subparsers.add_parser('build-train', help='Build train/val CSVs compatible with train.py.')
    build_parser.add_argument('--axisless-root', type=Path, required=True, help='Root directory containing axis-free trajectory images.')
    build_parser.add_argument('--raw-csv', type=Path, required=True, help='CSV with per-participant match statistics (e.g., raw.csv).')
    build_parser.add_argument('--output-dir', type=Path, default=Path('datasets'), help='Destination directory for train/val CSVs.')
    build_parser.add_argument('--num-images', type=int, default=DEFAULT_NUM_IMAGES, help='Number of participant images per match to include.')
    build_parser.add_argument('--train-ratio', type=float, default=0.8, help='Fraction of matches assigned to the training split.')
    build_parser.add_argument('--seed', type=int, default=42, help='Random seed for deterministic splitting.')
    build_parser.add_argument('--max-matches', type=int, default=None, help='Limit the total number of matches before splitting (useful for quick experiments).')

    minute_parser = subparsers.add_parser('build-minute', help='Build per-minute CSVs aligned with LolMinuteDataset expectations.')
    minute_parser.add_argument('--timeline-dir', type=Path, required=True, help='Directory containing timeline JSON files.')
    minute_parser.add_argument('--image-root', type=Path, required=True, help='Root directory containing per-minute trajectory images (e.g., image_dataset).')
    minute_parser.add_argument('--output-dir', type=Path, required=True, help='Destination root for the per-minute CSV hierarchy.')
    minute_parser.add_argument('--raw-csv', type=Path, required=True, help='Raw participant CSV with match outcomes for labels (e.g., raw.csv).')
    minute_parser.add_argument('--num-players', type=int, default=10, help='Number of participant PNGs expected per minute.')
    minute_parser.add_argument('--max-matches', type=int, default=500, help='Maximum number of matches to include.')
    minute_parser.add_argument('--details-dir', type=Path, default=None, help='Optional directory with match detail JSON files.')
    minute_parser.add_argument('--overview-csv', type=Path, default=None, help='Optional CSV providing participant champion overrides.')

    return parser, parser.parse_args(argv)

def process_input_dir(input_dir: str, output_root: str, details_dir: str = None, overview_csv: str = None) -> List[str]:
    os.makedirs(output_root, exist_ok=True)
    overview_df = build_overview_index(overview_csv) if overview_csv else None
    json_files = sorted(glob.glob(os.path.join(input_dir, "*.json")))
    if not json_files:
        print(f"[WARN] No .json files found in: {input_dir}")
    all_paths: List[str] = []
    for jp in json_files:
        try:
            parsed = parse_match_timeline(jp)
            per_minute = build_cumulative_tables(parsed["match_id"], parsed["frames"], parsed["puuids"],
                                                 details_dir, overview_df)
            out_dir = os.path.join(output_root, os.path.splitext(os.path.basename(jp))[0])
            paths = write_per_minute_csvs(parsed["match_id"], per_minute, out_dir)
            print(f"[OK] {jp} -> {len(paths)} CSVs at {out_dir}")
            all_paths.extend(paths)
        except Exception as e:
            print(f"[ERROR] Failed on {jp}: {e}")
    return all_paths


def main(argv: Optional[List[str]] = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    known_commands = {'per-minute', 'build-train', 'build-minute'}
    if argv and argv[0] not in known_commands and not argv[0].startswith('-'):
        if len(argv) < 2:
            print("Usage: python data_processing.py <timeline_dir> <output_root> [details_dir] [overview_csv]")
            return 1
        input_dir = argv[0]
        output_root = argv[1]
        details_dir = argv[2] if len(argv) >= 3 and os.path.isdir(argv[2]) else None
        overview_csv = None
        if len(argv) >= 3 and os.path.isfile(argv[2]) and argv[2].lower().endswith('.csv'):
            overview_csv = argv[2]
        if len(argv) >= 4:
            if os.path.isdir(argv[2]):
                details_dir = argv[2]
            if os.path.isfile(argv[3]) and argv[3].lower().endswith('.csv'):
                overview_csv = argv[3]
        process_input_dir(input_dir, output_root, details_dir, overview_csv)
        return 0
    parser, args = parse_cli_args(argv)
    if not getattr(args, 'command', None):
        parser.print_help()
        return 0
    if args.command == 'per-minute':
        process_input_dir(args.timeline_dir, args.output_root, args.details_dir, args.overview_csv)
        return 0
    if args.command == 'build-train':
        summary = build_training_dataset(
            args.axisless_root,
            args.raw_csv,
            args.output_dir,
            args.num_images,
            args.train_ratio,
            args.seed,
            args.max_matches,
        )
        print(f"Matches with valid image sets: {summary['matches_with_images']:,}")
        print(f"Matches with feature rows: {summary['matches_with_features']:,}")
        if summary['requested_max_matches'] is not None:
            print(f"Requested max matches: {summary['requested_max_matches']:,}")
        print(f"Matches in final dataset: {summary['matches_in_dataset']:,}")
        print(f"Train rows: {summary['train_rows']:,} -> {summary['train_path']}")
        print(f"Val rows:   {summary['val_rows']:,} -> {summary['val_path']}")
        return 0
    if args.command == 'build-minute':
        summary = build_minute_dataset(
            args.timeline_dir,
            args.image_root,
            args.output_dir,
            args.raw_csv,
            args.num_players,
            args.max_matches,
            args.details_dir,
            args.overview_csv,
        )
        print(f"Matches written: {summary['matches_written']:,}")
        print(f"Minutes written: {summary['minutes_written']:,}")
        print(f"Matches skipped (no images): {summary['matches_skipped_images']:,}")
        if summary['max_matches'] is not None:
            print(f"Requested max matches: {summary['max_matches']:,}")
        print(f"Output directory: {summary['output_dir']}")
        return 0
    parser.print_help()
    return 0

if __name__ == "__main__":
    raise SystemExit(main())