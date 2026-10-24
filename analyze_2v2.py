"""Analyse locale des replays Rocket League 2v2 et export par rang individuel.

Installation: uv pip install sprocket-rl-parser pandas numpy
Exemple: uv run python analyze_2v2.py --replays dataset/2v2 --output analyses_2v2
"""
import argparse
import csv
import json
import logging
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

RANK_ORDER = ['bronze','silver','gold','platinum','diamond','champion','grand_champion','supersonic_legend']
METRICS = ['goals','shots','saves','assists','score','speed_mean_uu_s','speed_max_uu_s',
           'distance_traveled_uu','distance_to_ball_mean_uu','distance_to_teammate_mean_uu',
           'height_mean_uu','airtime_percent','time_own_half_percent','boost_mean',
           'boost_low_percent','boost_spent_estimated']

def number(v):
    if isinstance(v, dict):
        return None
    try:
        return float(v) if v is not None and str(v).strip() else None
    except (TypeError, ValueError):
        return None

def slug(v):
    return re.sub(r'[^a-z0-9]+','_',str(v or '').lower()).strip('_')

def rank_info(player):
    rank = player.get('rank') or {}
    if isinstance(rank, str):
        rank = {'name':rank}
    name = rank.get('name') or player.get('rank_name') or ''
    rid = rank.get('id') or ''
    s = slug(f'{rid} {name}').replace('grand_champion','grandchampion')
    if 'supersonic_legend' in s or 'supersoniclegend' in s or re.search(r'\bssl\b',s):
        group = 'supersonic_legend'
    elif 'grandchampion' in s:
        group = 'grand_champion'
    else:
        group = next((r for r in RANK_ORDER if r in s), 'unknown')
    return {'rank_group':group, 'rank_name':name or rid or 'Unknown',
            'rank_id':rid, 'rank_tier':rank.get('tier'), 'rank_division':rank.get('division')}

def get_players(meta):
    # Le collecteur conserve les joueurs dans data.blue / data.orange.
    deep = meta.get('data') if isinstance(meta.get('data'),dict) else meta
    result = []
    for team in ('blue','orange'):
        team_obj = deep.get(team) or meta.get(team) or {}
        for p in team_obj.get('players',[]):
            if isinstance(p,dict):
                result.append((team,p))
    # Secours : champ player_ranks du collecteur personnalisé
    if not result:
        for p in meta.get('player_ranks',[]):
            if isinstance(p,dict):
                rank = {'name':p.get('rank_name'),'id':p.get('rank_id'),
                        'tier':p.get('rank_tier'),'division':p.get('rank_division')}
                result.append((p.get('team','unknown'), {**p,'name':p.get('player_name'), 'rank':rank}))
    return result

def load_metadata(path):
    result = {}
    if not path.exists():
        return result
    with path.open(encoding='utf-8-sig') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                obj = json.loads(line)
                rid = obj.get('id') or (obj.get('data') or {}).get('id')
                if rid: result[rid] = obj
            except json.JSONDecodeError:
                logging.warning('Ligne JSON invalide ignorée dans %s', path)
    return result

def extract_stat(p, key):
    # Ballchasing: player.stats.core.goals, player.stats.core.shots, etc.
    stats = p.get('stats') or {}
    core = stats.get('core') or {}
    return number(core.get(key, p.get(key)))

def get_frame_table(df, name):
    if not isinstance(df.columns,pd.MultiIndex):
        return None
    if name in df.columns.get_level_values(0):
        return df[name]
    matches = [x for x in df.columns.get_level_values(0).unique() if str(x).strip().casefold() == str(name).strip().casefold()]
    return df[matches[0]] if len(matches)==1 else None

def xyz(tab):
    if tab is None or not all(k in tab.columns for k in ('pos_x','pos_y','pos_z')):
        return None
    return tab[['pos_x','pos_y','pos_z']].apply(pd.to_numeric,errors='coerce')

def series(tab, candidates):
    if tab is None: return None
    for k in candidates:
        if k in tab.columns:
            return pd.to_numeric(tab[k], errors='coerce')
    return None

def mean_or_none(s):
    if s is None: return None
    v = pd.to_numeric(s,errors='coerce').dropna()
    return float(v.mean()) if len(v) else None

def compute_frames(df, name, mate_name, team):
    out = {k:None for k in METRICS if k not in ('goals','shots','saves','assists','score')}
    car = get_frame_table(df,name)
    if car is None: return out
    pos = xyz(car)
    ball = xyz(get_frame_table(df,'ball'))
    mate = xyz(get_frame_table(df,mate_name)) if mate_name else None
    game = get_frame_table(df,'game')
    t = series(game, ('time','seconds','game_time'))
    if t is not None:
        dt = t.diff().where(lambda x:(x>0)&(x<1),np.nan)
    else:
        dt = None
    speed = series(car,('speed',))
    if speed is None and all(k in car.columns for k in ('vel_x','vel_y','vel_z')):
        vel = car[['vel_x','vel_y','vel_z']].apply(pd.to_numeric,errors='coerce')
        speed = np.sqrt((vel**2).sum(axis=1,min_count=3))
    if speed is not None:
        speed = speed.replace([np.inf, -np.inf], np.nan)
        valid = speed.dropna()
        # Certains exports carball donnent les vitesses en dixiemes d'uu/s.
        # La detection est volontairement conservatrice : en cas d'ambiguite,
        # mieux vaut une valeur absente qu'une statistique fausse.
        if len(valid):
            p95 = float(valid.quantile(0.95))
            vmax = float(valid.max())
            if 3000 < p95 <= 30000 and vmax <= 35000:
                speed = speed / 10.0
            elif p95 > 3000 or vmax > 35000:
                speed = None
            if speed is not None:
                out['speed_mean_uu_s'] = mean_or_none(speed)
                out['speed_max_uu_s'] = float(speed.max()) if speed.notna().any() else None
    if pos is not None:
        dif = pos.diff()
        step = np.sqrt((dif**2).sum(axis=1,min_count=3))
        # Écarte les téléportations / respawns et grands sauts de coordonnées.
        step = step.where(step<1500)
        # Une distance mesuree entre frames est independante de l'echelle
        # des colonnes de vitesse; elle peut toutefois manquer des frames.
        out['distance_traveled_uu'] = float(step.sum()) if step.notna().any() else None
        out['height_mean_uu'] = mean_or_none(pos['pos_z'])
        out['airtime_percent'] = mean_or_none((pos['pos_z']>250).where(pos['pos_z'].notna()).astype(float))*100
        if ball is not None:
            d = np.sqrt(((pos-ball)**2).sum(axis=1,min_count=3))
            out['distance_to_ball_mean_uu'] = mean_or_none(d)
        if mate is not None:
            d = np.sqrt(((pos-mate)**2).sum(axis=1,min_count=3))
            out['distance_to_teammate_mean_uu'] = mean_or_none(d)
        # Orientation du terrain: bleu défend y<0, orange y>0 (convention RL).
        own = (pos['pos_y']<0) if team=='blue' else (pos['pos_y']>0)
        out['time_own_half_percent'] = mean_or_none(own.where(pos['pos_y'].notna()).astype(float))*100
    boost = series(car,('boost','boost_amount','boost_level'))
    if boost is not None:
        boost = boost.replace([np.inf, -np.inf], np.nan)
        observed = boost.dropna()
        # Une colonne constante a zero sur un match entier est souvent
        # une colonne non renseignee. Ne pas la traiter comme une mesure.
        if len(observed) >= 10 and observed.nunique() > 1 and observed.max() > 0:
            # Accepter uniquement une echelle explicite 0..100.
            if observed.min() >= 0 and observed.max() <= 100:
                out['boost_mean'] = mean_or_none(boost)
                out['boost_low_percent'] = mean_or_none((boost < 20).where(boost.notna()).astype(float)) * 100
                dif = boost.diff()
                spent = -dif.where(dif < 0)
                out['boost_spent_estimated'] = float(spent.sum()) if spent.notna().any() else None
    return out

def write_csv(path,rows,fields):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore')
        w.writeheader();w.writerows(rows)

def main():
    ap=argparse.ArgumentParser(description='Statistiques Rocket League 2v2 par rang individuel')
    ap.add_argument('--replays',type=Path,default=Path('dataset/2v2'))
    ap.add_argument('--metadata',type=Path,default=None)
    ap.add_argument('--output',type=Path,default=Path('analyses_2v2'))
    ap.add_argument('--limit',type=int,default=0,help='Nombre de replays à tester, 0=tous')
    ap.add_argument('--metadata-only',action='store_true',help='Sans parseur spatial')
    args=ap.parse_args()
    logging.basicConfig(level=logging.INFO,format='%(levelname)s: %(message)s')
    metadata=load_metadata(args.metadata or args.replays/'metadata.json')
    paths=sorted(args.replays.glob('*.replay'))
    if args.limit: paths=paths[:args.limit]
    if not paths:
        ap.error(f'Aucun fichier .replay dans {args.replays}')
    if not args.metadata_only:
        try: import carball
        except ImportError:
            ap.error('Parseur absent. Installe-le : uv pip install sprocket-rl-parser pandas numpy')
    rows=[]; errors=[]
    for i,path in enumerate(paths,1):
        rid=path.stem
        meta=metadata.get(rid)
        if not meta:
            errors.append({'replay_id':rid,'error':'Métadonnées absentes : rang inconnu'})
            continue
        players=get_players(meta)
        if len(players)!=4 or Counter(t for t,_ in players)!=Counter({'blue':2,'orange':2}):
            errors.append({'replay_id':rid,'error':'Métadonnées joueurs 2v2 incomplètes'})
            continue
        df=None
        if not args.metadata_only:
            try:
                manager=carball.analyze_replay_file(str(path),calculate_intensive_events=False)
                df=manager.get_data_frame()
            except Exception as e:
                errors.append({'replay_id':rid,'error':f'Parsing spatial impossible : {type(e).__name__}: {e}'})
        for team,p in players:
            name=p.get('name') or p.get('player_name') or ''
            mates=[q.get('name') for t,q in players if t==team and q is not p]
            r=rank_info(p)
            pid=p.get('id') or {}
            if isinstance(pid, dict):
                pid=f"{pid.get('platform', 'unknown')}:{pid.get('id', '')}"
            row={'replay_id':rid,'player_name':name,'player_id':str(pid),
                 'team':team,**r,'spatial_available':False}
            for k in ('goals','shots','saves','assists','score'):
                row[k]=extract_stat(p,k)
            if df is not None:
                spatial=compute_frames(df,name,mates[0] if mates else None,team)
                row.update(spatial)
                row['spatial_available']=any(v is not None for v in spatial.values())
            rows.append(row)
        print(f'[{i}/{len(paths)}] {rid} : {len(players)} joueurs',flush=True)
    if not rows:
        ap.error('Aucun joueur extrait. Vérifie metadata.json et le dossier des replays.')
    fields=['replay_id','player_name','player_id','team','rank_group','rank_name','rank_id',
            'rank_tier','rank_division','spatial_available']+METRICS
    write_csv(args.output/'players_all.csv',rows,fields)
    # Signaler explicitement les valeurs manquantes pour ne pas les
    # confondre avec des zeros reels dans les analyses ulterieures.
    spatial_missing = sum(not row['spatial_available'] for row in rows)
    boost_missing = sum(row.get('boost_mean') is None for row in rows)
    logging.info('Spatial absent : %d/%d ; boost non fiable/absent : %d/%d',
                 spatial_missing, len(rows), boost_missing, len(rows))
    grouped=defaultdict(list)
    for row in rows: grouped[row['rank_group']].append(row)
    for group,items in grouped.items():
        write_csv(args.output/'by_rank'/f'{group}.csv',items,fields)
    summary=[]
    for group,items in sorted(grouped.items()):
        record={'rank_group':group,'observations':len(items),
                'distinct_player_ids':len(set(x['player_id'] for x in items if x['player_id'])),
                'replays':len(set(x['replay_id'] for x in items)),
                'spatial_observations':sum(bool(x['spatial_available']) for x in items)}
        for k in METRICS:
            vals=[number(x.get(k)) for x in items]
            vals=[x for x in vals if x is not None and math.isfinite(x)]
            record[f'{k}_mean']=round(sum(vals)/len(vals),3) if vals else None
            record[f'{k}_n']=len(vals)
        summary.append(record)
    sumfields=['rank_group','observations','distinct_player_ids','replays','spatial_observations']
    for k in METRICS: sumfields.extend([f'{k}_mean',f'{k}_n'])
    write_csv(args.output/'averages_by_rank.csv',summary,sumfields)
    write_csv(args.output/'errors.csv',errors,['replay_id','error'])
    print(f'\nTerminé : {len(rows)} observations, {len(grouped)} groupes de rang, {len(errors)} avertissements')
    print('Résultats :',args.output.resolve())
    for s in summary: print(f"  {s['rank_group']}: {s['observations']} observations, {s['spatial_observations']} avec spatial")

if __name__=='__main__': main()
