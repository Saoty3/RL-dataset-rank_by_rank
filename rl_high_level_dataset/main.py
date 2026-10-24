import argparse
import glob
import json
import logging
import math
import os
import shelve
import shutil
import tempfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Iterator, Optional, Iterable, Set

import requests
import ballchasing as bc
from ballchasing.util import get_players, get_pid, is_standard_replay, ensure_sorted, deduplicate, mix_replay_iterators, \
    get_gameplay_duration
from requests import HTTPError
from requests.adapters import HTTPAdapter
from tqdm import tqdm

from rl_high_level_dataset.anomaly import is_anomalous_replay
from rl_high_level_dataset.encounters import get_encounter_stats, EncounterStats
from rl_high_level_dataset.whosbotting import (
    has_cheater,
    load_whosbotting_cache,
    check_cached_cheater,
    load_cheater_accounts,
    replay_has_known_cheater,
)

REPLAYS_PER_SEASON = int(os.getenv("REPLAYS_PER_SEASON", "1000"))


def filter_valid_replays(replays: Iterator[dict]) -> Iterator[dict]:
    for replay in replays:
        is_standard, reason = is_standard_replay(replay)
        if is_standard:
            yield replay
        else:
            logging.info(
                f"Skipping replay {replay.get('id')} due to: {reason}"
            )


def score_and_filter_replays(replays: Iterator[dict], player_scores: dict,
                             threshold: float = 0.25,
                             cheater_accounts: Optional[Set[str]] = None) -> Iterator[tuple[dict, float]]:
    for replay in replays:
        if cheater_accounts and replay_has_known_cheater(replay, cheater_accounts):
            logging.debug(f"Skipping replay {replay.get('id')} due to known cheater account.")
            continue
        players = get_players(replay)
        if not players:
            continue
        tot = 0.0
        for player in players:
            pid = get_pid(player)
            if not pid or pid not in player_scores:
                logging.debug(
                    f"Skipping replay {replay.get('id')} due to player {pid} not being qualified."
                )
                break
            ps = player_scores[pid]
            if ps.prob_pro <= 0 or ps.prob_ssl <= 0:
                break
            tot += math.log(ps.prob_pro) + math.log(ps.prob_ssl)
        else:
            # Probability that players are both pro and SSL
            score = math.exp(tot / len(players))  # Geometric mean
            if score > threshold:
                yield replay, score


def iterate_replays_cached(
        replays: Optional[Iterator[dict]],
        cache_path: str,
        overwrite: bool = False,
        append: bool = False,
) -> Iterator[tuple[dict, bool]]:
    # If the cache file exists, and we aren't overwriting or appending, just read and return
    if os.path.exists(cache_path) and not overwrite and not append:
        with open(cache_path, "r") as reader:
            for line in reader:
                line = line.strip()
                if line:
                    yield json.loads(line), True
        return

    # Create the cache directory and prepare the temporary working file
    working_path = cache_path.replace(".jsonl", "_working.jsonl")
    assert working_path != cache_path, "Temporary file must not be the same as the original file"
    cache_dir = os.path.dirname(cache_path)
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)

    mode = "a+" if append and not overwrite else "w"
    seen = set()
    with open(working_path, mode) as writer:
        if mode == "a+":
            # Read the existing cache and yield it
            writer.seek(0)
            for line in writer:
                line = line.strip()
                if not line:
                    continue
                replay = json.loads(line)
                yield replay, True
                seen.add(replay["id"])

        # Now write the new incoming replays
        if replays is not None:
            for replay in replays:
                if replay["id"] not in seen:
                    writer.write(f"{json.dumps(replay)}\n")
                    seen.add(replay["id"])
                    yield replay, False

    # Atomically replace the old cache with the new working file
    if os.path.exists(cache_path):
        os.remove(cache_path)
    os.rename(working_path, cache_path)


def shared_pipeline(replays: Iterator[dict], cache_path: str, result_path: Optional[str] = None) -> Iterator[dict]:
    # First cache of raw results
    replays = iterate_replays_cached(replays, cache_path)
    replays = (replay for replay, from_cache in replays)
    # Make sure the replays are valid, sorted and unique
    replays = filter_valid_replays(replays)
    replays = ensure_sorted(replays, sort_dir=bc.SortDir.ASCENDING, sort_by=bc.ReplaySortBy.REPLAY_DATE)
    replays = deduplicate(replays, check_dates=True)
    # Second cache for results
    if result_path:
        replays = iterate_replays_cached(replays, result_path)
        replays = (replay for replay, from_cache in replays)
    yield from replays


def get_ranked_replays(bc_api: bc.Api, season: str, cache_dir: str):
    """Recherche des matchs classés 2v2, sans rang minimum."""
    ranked_replays = bc_api.get_replays(
        playlist=bc.Playlist.RANKED_DOUBLES,
        sort_by=bc.ReplaySortBy.REPLAY_DATE,
        sort_dir=bc.SortDir.ASCENDING,
        count=REPLAYS_PER_SEASON,
        season=season,
    )
    ranked_cache_path = os.path.join(cache_dir, f"season_{season}", "ranked", "all_ranks_2v2.jsonl")
    ranked_result_path = ranked_cache_path.replace(".jsonl", "_results.jsonl")
    yield from shared_pipeline(ranked_replays, ranked_cache_path, ranked_result_path)


def get_private_replays(bc_api: bc.Api, season: str, player_id: str, cache_dir: str):
    private_replays = bc_api.get_replays(
        playlist=[bc.Playlist.PRIVATE, bc.Playlist.OFFLINE],
        sort_by=bc.ReplaySortBy.REPLAY_DATE,
        sort_dir=bc.SortDir.ASCENDING,
        season=season,
        count=REPLAYS_PER_SEASON,
        player_id=player_id,
        disable_prefetch=True,  # Prevent way too many requests at once since we do this for each player
    )
    clean_pid = player_id.replace(":", "_").replace("*", "x")
    private_cache_path = os.path.join(cache_dir, f"season_{season}", "private", f"{clean_pid}.jsonl")
    private_result_path = private_cache_path.replace(".jsonl", "_results.jsonl")
    private_replays = shared_pipeline(private_replays, private_cache_path, private_result_path)
    yield from private_replays


def get_qualified_player_scores(
        encounter_stats: dict[str, EncounterStats],
        player_stats: dict[str, EncounterStats],
        or_threshold: float = 0.0,
        and_threshold: float = 0.0,
        cheater_accounts: Optional[Set[str]] = None,
) -> dict[str, EncounterStats]:
    """Accepte tous les joueurs observés, sans test pro ou SSL."""
    player_scores = {}

    for pid, encounters in encounter_stats.items():
        if cheater_accounts and pid in cheater_accounts:
            continue

        own = player_stats.get(pid, EncounterStats())
        player_scores[pid] = encounters - own

    return player_scores

def filter_ranked_2v2_replays(replays):
    """Conserve les matchs 2v2 dont les 4 joueurs ont un rang connu."""

    for replay in replays:
        if replay.get("playlist_id") != "ranked-doubles":
            continue

        blue_players = replay.get("blue", {}).get("players", [])
        orange_players = replay.get("orange", {}).get("players", [])

        players = blue_players + orange_players

        if len(blue_players) != 2 or len(orange_players) != 2:
            continue

        if not all(
            (player.get("rank") or {}).get("id")
            not in (None, "unranked")
            for player in players
        ):
            continue

        yield replay, 1.0

def extract_player_ranks(replay):
    result = []

    for team in ("blue", "orange"):
        for player in replay.get(team, {}).get("players", []):
            rank = player.get("rank") or {}

            result.append({
                "replay_id": replay["id"],
                "player_name": player.get("name"),
                "player_id": player.get("id"),
                "team": team,
                "rank_id": rank.get("id"),
                "rank_name": rank.get("name"),
                "rank_tier": rank.get("tier"),
                "rank_division": rank.get("division"),
                "score": player.get("score"),
            })

    return result

def collect_replays(bc_api: bc.Api, cache_dir: str, cheater_accounts: Optional[Set[str]] = None) -> Iterator[tuple[dict, float]]:
    """Collecte les matchs 2v2 avec rang connu, sans qualification pro/SSL."""
    seasons = ["f21"]
    seen = set()
    for season in seasons:
        logging.critical(f"Collecting ranked 2v2 replays for season {season}...")
        count = 0
        for replay, score in filter_ranked_2v2_replays(get_ranked_replays(bc_api, season, cache_dir)):
            rid = replay.get("id")
            if not rid or rid in seen:
                continue
            if cheater_accounts and replay_has_known_cheater(replay, cheater_accounts):
                continue
            seen.add(rid)
            count += 1
            yield replay, score
        logging.critical(f"Collected {count} ranked 2v2 replays with 4 known ranks for season {season}.")


def collect_replay_scores(bc_api: bc.Api, cache_dir: str, cheater_accounts: Optional[Set[str]] = None):
    # Collect replay scores
    scores_path = os.path.join(cache_dir, "scores.json")
    if os.path.exists(scores_path):
        with open(scores_path, "r") as f:
            scores = json.load(f)
        scores = {mode: data for mode, data in scores.items() if mode in ("1v1", "2v2", "3v3")}
        if cheater_accounts:
            removed_count = 0
            for mode in list(scores.keys()):
                for rid, rinfo in list(scores[mode].items()):
                    replay_players = set(rinfo.get("players", []))
                    if replay_players & cheater_accounts:
                        del scores[mode][rid]
                        removed_count += 1
            if removed_count:
                logging.critical(f"Filtered out {removed_count} cheater replays from cached scores.")
    else:
        scores = {}
        replays = collect_replays(bc_api, cache_dir, cheater_accounts=cheater_accounts)
        for replay, score in replays:
            players = get_players(replay)
            gameplay_duration = get_gameplay_duration(replay)
            mode = f"{len(players) // 2}v{len(players) // 2}"
            if mode not in ("1v1", "2v2", "3v3"):
                continue
            rid = replay["id"]
            scores.setdefault(mode, {})[rid] = {
                "id": rid,
                "score": score,
                "gameplay_duration": gameplay_duration,
                "players": [get_pid(player) for player in players],
                "player_ranks": extract_player_ranks(replay),
            }
        with open(scores_path, "w") as f:
            json.dump(scores, f, indent=2)
    return scores


def select_replays(scores: dict, max_replays: int = 100) -> Iterator[str]:
    """Sélectionne au maximum 100 replays 2v2, sans score de niveau pro."""
    available = list(scores.get("2v2", {}).values())
    if not available:
        logging.warning("Aucun replay 2v2 avec quatre rangs connus.")
        return
    for replay_info in available[:max_replays]:
        yield replay_info["id"]


def get_deep_replays(
        bc_api: bc.Api,
        replays: Iterable[str],
        shelf_path: str,
        workers: int = 4
) -> Iterator[dict]:
    # Open the shelf initially to identify which replays are missing
    with shelve.open(shelf_path) as shelf:
        uncached_replays = [rid for rid in replays if rid not in shelf]

    # Create an iterator over the uncached replays using tqdm to display progress
    it = tqdm(
        uncached_replays,
        desc="Fetching uncached deep replays",
        total=len(uncached_replays),
        unit="replay"
    )

    def fetch_replay(replay_id: str) -> tuple[str, Optional[dict]]:
        try:
            return replay_id, bc_api.get_replay(replay_id)
        except requests.HTTPError as e:
            if e.response.status_code == 404:
                return replay_id, None
            raise

    # Download missing replays concurrently and save them as they complete
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for replay_id, replay in executor.map(fetch_replay, it):
            if replay is not None:
                # Open, write, and immediately close the shelf for each item to avoid corrupting it on exit
                with shelve.open(shelf_path) as shelf:
                    shelf[replay_id] = replay
            it.set_postfix(dict(id=replay_id, rate_limits=bc_api.rate_limit_count))

    # Finally, read and yield all the replays sequentially from the shelf
    with shelve.open(shelf_path) as shelf:
        for rid in replays:
            if rid in shelf:
                yield shelf[rid]


def main():
    cur_path = os.path.abspath(os.path.dirname(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", type=str, default=os.path.join(cur_path, "..", "out", "high_level"))
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument("--whosbotting-cache", type=str, default=None)
    parser.add_argument("--enable-whosbotting", action="store_true", default=False,
                        help="Enable ML-based bot detection via whosbotting.com (currently disabled by default)")
    parser.add_argument("--cheaters-file", type=str, default=None)
    parser.add_argument("--disable-anomaly-filter", action="store_true", default=False,
                        help="Disable statistical and behavioral anomaly filtering for non-serious / AFK matches.")
    parser.add_argument("--out-path", type=str)
    args = parser.parse_args()

    api_key = os.getenv("BC_API_KEY")
    assert api_key, "BC_API_KEY not set"
    bc_api = bc.Api(api_key, proactive_rate_limit=True, print_on_rate_limit=True)
    bc_api._session.mount("https://", HTTPAdapter(pool_connections=100, pool_maxsize=100))
    base_dir = Path(args.base_dir)
    cache_dir = Path(args.cache_dir) if args.cache_dir else base_dir / "cache"
    out_path = Path(args.out_path)

    base_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_path.mkdir(parents=True, exist_ok=True)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARN)
    file_handler = logging.FileHandler(str(base_dir / "collect_replays.log"), mode="w")
    file_handler.setLevel(logging.DEBUG)
    logging.basicConfig(
        handlers=[console_handler, file_handler],
        level=logging.NOTSET,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Load known cheater accounts list (temporary workaround for whosbotting)
    cheaters_file = Path(args.cheaters_file) if args.cheaters_file else Path(cur_path).parent / "cheaters.txt"
    cheater_accounts = load_cheater_accounts(cheaters_file)
    if cheater_accounts:
        logging.critical(f"Loaded {len(cheater_accounts)} known cheater accounts from {cheaters_file}")

    logging.critical("Starting replay collection...")
    scores = collect_replay_scores(bc_api, str(cache_dir), cheater_accounts=cheater_accounts)
    logging.critical("Scores calculated.")
    logging.info(f"Ballchasing rate limit stats after scoring: {bc_api.rate_limit_stats}")

    selected_replays = list(select_replays(scores))
    logging.critical(f"Selected {len(selected_replays)} replays.")
    shelf_path = str(cache_dir / "deep_replays.shelve")

    # Get detailed versions of the selected replays, and re-filter with new info
    deep_replays = get_deep_replays(bc_api, selected_replays, shelf_path, workers=3)
    deep_replays = filter_valid_replays(deep_replays)
    deep_replays = deduplicate(deep_replays, check_dates=False)  # Just check IDs now that MatchGUID is available
    replay_ids = set()
    with tqdm(deep_replays, desc="Collecting deep replays.", total=len(selected_replays), unit="replay") as it:
        for replay in it:
            replay_ids.add(replay["id"])
            it.set_postfix(dict(id=replay["id"], date=replay["date"], rate_limits=bc_api.rate_limit_count))

    logging.info(f"Ballchasing rate limit stats after deep replays: {bc_api.rate_limit_stats}")

    # Re-filter any deep replays containing known cheater accounts
    if cheater_accounts:
        with shelve.open(shelf_path) as shelf:
            flagged = {
                rid for rid in replay_ids
                if rid in shelf and replay_has_known_cheater(shelf[rid], cheater_accounts)
            }
            if flagged:
                logging.critical(f"Filtered out {len(flagged)} deep replays matching accounts in {cheaters_file}")
                replay_ids -= flagged

    # Filter out anomalous replays (AFK, freestyle, non-serious, mutator lobbies)
    if not args.disable_anomaly_filter:
        with shelve.open(shelf_path) as shelf:
            anomalous = set()
            for mode, mode_scores in scores.items():
                for rid in mode_scores:
                    if rid in replay_ids and rid in shelf:
                        is_anom, reasons = is_anomalous_replay(shelf[rid], mode=mode)
                        if is_anom:
                            anomalous.add(rid)
                            logging.debug(f"Filtered anomalous replay {rid}: {reasons}")
            if anomalous:
                logging.critical(f"Filtered out {len(anomalous)} anomalous replays across modes.")
                replay_ids -= anomalous

    # New scores with valid replays, then rebalance
    scores = {
        mode: {rid: rinfo for rid, rinfo in mode_scores.items() if rid in replay_ids}
        for mode, mode_scores in scores.items()
    }
    selected_replays = list(select_replays(scores))
    replay_ids = set(selected_replays)

    logging.critical(f"Collected {len(replay_ids)} deep replays.")

    # Load cached whosbotting verdicts if file exists
    whosbotting_cache_path = Path(args.whosbotting_cache) if args.whosbotting_cache else cache_dir / "whosbotting.jsonl"
    whosbotting_cache = load_whosbotting_cache(whosbotting_cache_path) if whosbotting_cache_path.exists() else {}
    if whosbotting_cache:
        logging.critical(f"Loaded {len(whosbotting_cache)} cached whosbotting verdicts from {whosbotting_cache_path}")

    # Download replays and filter out cheaters in a single pass
    # Storage budget for the dataset
    MAX_STORAGE_BYTES = 8 * 1024**3  # 8 GiB
    cheater_replays = set()
    with tempfile.TemporaryDirectory() as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        for mode in scores:
            mode_path = out_path / mode
            mode_path.mkdir(parents=True, exist_ok=True)
            existing = {
                p.stem for p in mode_path.glob("*.replay")
            }
            to_delete = set()  # Ne pas supprimer les anciens fichiers lors des tests
            logging.info(f"Deleting {len(to_delete)} replays that are no longer needed for mode {mode}.")
            for rid in to_delete:
                (mode_path / f"{rid}.replay").unlink(missing_ok=True)

            to_include = [rinfo for rinfo in scores[mode].values() if rinfo["id"] in replay_ids]
            for replay_info in tqdm(to_include, desc=f"Downloading replays for mode {mode}"):
                rid = replay_info["id"]
                replay_path = mode_path / f"{rid}.replay"

                # Check if already cached as cheater
                cached_is_cheater = check_cached_cheater(rid, cache=whosbotting_cache)
                if cached_is_cheater is True:
                    cheater_replays.add(rid)
                    (mode_path / f"{rid}.replay").unlink(missing_ok=True)
                    continue

                # If already on disk: keep clean, only send to whosbotting if explicitly enabled
                if replay_path.exists():
                    if args.enable_whosbotting and cached_is_cheater is not False:
                        if has_cheater(replay_path, cache_path=whosbotting_cache_path, replay_id=rid, cache=whosbotting_cache):
                            cheater_replays.add(rid)
                            replay_path.unlink(missing_ok=True)
                    continue

                # Replay not yet downloaded: download to temp
                used_bytes = sum(
                    p.stat().st_size
                    for p in out_path.rglob("*")
                    if p.is_file()
                )

                if used_bytes >= MAX_STORAGE_BYTES:
                    raise RuntimeError(
                        f"Storage budget reached: {used_bytes / 1024**3:.2f} GiB. "
                        "Stopping downloads to protect disk space."
                    )

                # Keep at least 18 GiB free on the system drive
                MIN_FREE_BYTES = 18 * 1024**3

                free_bytes = shutil.disk_usage(tmp_dir).free

                if free_bytes < MIN_FREE_BYTES:
                    raise RuntimeError(
                        f"Insufficient disk space: "
                        f"{free_bytes / 1024**3:.2f} GiB remaining. "
                        "Download stopped."
                    )

                tmp_path = tmp_dir / f"{rid}.replay"

                bc_api.download_replay(rid, tmp_path)

                # Check budget including the newly downloaded file
                if used_bytes + tmp_path.stat().st_size > MAX_STORAGE_BYTES:
                    tmp_path.unlink(missing_ok=True)
                    raise RuntimeError(
                        "Storage budget exceeded by next replay. "
                        "Download stopped."
                    )
                if args.enable_whosbotting and cached_is_cheater is not False:
                    if has_cheater(tmp_path, cache_path=whosbotting_cache_path, replay_id=rid, cache=whosbotting_cache):
                        cheater_replays.add(rid)
                        tmp_path.unlink(missing_ok=True)
                        continue
                shutil.move(tmp_path, replay_path)

    # Remove cheaters from the final selected replay set and re-balance
    if cheater_replays:
        logging.critical(f"Filtered out {len(cheater_replays)} cheater replays.")
        replay_ids -= cheater_replays
        scores = {
            mode: {rid: rinfo for rid, rinfo in mode_scores.items() if rid in replay_ids}
            for mode, mode_scores in scores.items()
        }
        selected_replays = list(select_replays(scores))
        replay_ids = set(selected_replays)

    downloaded_replays = set(out_path.glob("*/*.replay"))
    # Clean up any downloaded replays that were dropped in re-balancing
    downloaded_replays = {p for p in downloaded_replays if p.stem in replay_ids}  # Ignorer les fichiers hors selection

    assert len(downloaded_replays) == len(replay_ids), (
        f"Mismatch between downloaded replays ({len(downloaded_replays)}) and selected IDs ({len(replay_ids)})!"
    )
    logging.critical(f"Replay download complete. {len(replay_ids)} replays verified. Creating metadata files.")

    # Finally, add metadata files with the included replays
    with shelve.open(shelf_path) as shelf:
        for mode in scores:
            to_include = list(rinfo for rinfo in scores[mode].values() if rinfo["id"] in replay_ids)
            metadata_path = out_path / mode / "metadata.json"
            with (open(metadata_path, "w") as f,
                  tqdm(to_include, desc=f"Creating metadata file for mode {mode}") as it):
                for replay_info in it:
                    rid = replay_info["id"]
                    deep = shelf[rid]
                    f.write(json.dumps({
                        **replay_info,
                        "data": deep,
                        "player_ranks": replay_info.get("player_ranks", []),
                    }) + "\n")

    logging.info(f"Final Ballchasing rate limit stats: {bc_api.rate_limit_stats}")
    logging.critical("Finished.")


if __name__ == '__main__':
    main()
