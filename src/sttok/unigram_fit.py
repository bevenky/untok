"""Native-bank-constrained Unigram research fitting.

The native SentencePiece vocabulary, piece messages and normalizer stay fixed.
Only appended NORMAL-piece scores are fitted. A caller-selected, fixed sum of
exp(appended_score) anchors the fit; it is not a globally normalized LM.

Input text is always RAW text, normalized exactly once with the native model.
For script-span research, pass the raw span as text; its native dummy prefix is
then real for that span. Full-record evaluation remains a separate requirement.
All corpus weights and candidate provenance are explicit. No downloads occur.
"""
from __future__ import annotations
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import heapq
import math
from typing import Any, Iterable
import sentencepiece as spm
from sentencepiece import sentencepiece_model_pb2 as pb

NEG = -math.inf
NORMAL = 1
UNKNOWN = 2
USER_DEFINED = 4


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _logadd(a: float, b: float) -> float:
    if a == NEG: return b
    if b == NEG: return a
    if b > a: a, b = b, a
    return a + math.log1p(math.exp(b - a))


def _float32(value: float) -> float:
    # Match the persisted SentencePiece float representation before comparisons.
    import struct
    return struct.unpack('f', struct.pack('f', value))[0]


def load_native(model: bytes | pb.ModelProto) -> pb.ModelProto:
    p = pb.ModelProto()
    if isinstance(model, bytes): p.ParseFromString(model)
    else: p.CopyFrom(model)
    if p.trainer_spec.model_type != pb.TrainerSpec.UNIGRAM:
        raise ValueError('Expected the actual native UNIGRAM model, not an HF BPE export')
    if len({x.piece for x in p.pieces}) != len(p.pieces):
        raise ValueError('Native model contains duplicate piece strings')
    return p


def build_model(native: pb.ModelProto, additions: dict[str, float]) -> pb.ModelProto:
    """Preserve every original piece message/ID and every normalizer byte."""
    model = pb.ModelProto(); model.CopyFrom(native)
    old = {p.piece for p in native.pieces}
    normals = [p.score for p in native.pieces if p.type == NORMAL]
    low, high = min(normals), max(normals)
    for piece, score in sorted(additions.items()):
        if piece in old:
            raise ValueError(f'Attempted to append native piece {piece!r}')
        if not piece or not math.isfinite(score) or not low <= score <= high:
            raise ValueError(f'Invalid appended piece/score {piece!r}: {score}')
        p = model.pieces.add(); p.piece = piece; p.score = score; p.type = NORMAL
    model.trainer_spec.vocab_size = len(model.pieces)
    assert all(a.SerializeToString() == b.SerializeToString()
               for a, b in zip(native.pieces, model.pieces))
    assert model.normalizer_spec.SerializeToString() == native.normalizer_spec.SerializeToString()
    if native.HasField('denormalizer_spec'):
        assert model.denormalizer_spec.SerializeToString() == native.denormalizer_spec.SerializeToString()
    return model


@dataclass
class Record:
    text: str                 # Native-normalized text, including synthetic prefix.
    weight: float
    language: str
    source: str
    raw_example: str


class Lattice:
    """Precompiled Unicode NORMAL-piece lattice, with native unknown fallback.

    UNKNOWN runs affect SP output-ID counts but not their per-character path
    energy. USER_DEFINED strings are rejected at intake, since their forced
    matching semantics are outside this normal-text fitting implementation.
    """
    def __init__(self, text: str, trie: dict, unknown_score: float):
        self.text = text
        self.edges: list[list[tuple[int, str | None]]] = [[] for _ in text]
        self.unknown_score = unknown_score
        for start in range(len(text)):
            node = trie; has_single = False
            for end in range(start, len(text)):
                node = node.get(text[end])
                if node is None: break
                piece = node.get(None)
                if piece is not None:
                    self.edges[start].append((end + 1, piece))
                    if end == start: has_single = True
            # Keep fallback compiled, activate it only when no CURRENT single
            # character piece exists. Pruning must not make fallback disappear.
            self.edges[start].append((start + 1, None))

    def expected(self, scores: dict[str, float], weight: float = 1.0):
        n = len(self.text); alpha = [NEG] * (n + 1); alpha[0] = 0.0
        for i, edges in enumerate(self.edges):
            if alpha[i] == NEG: continue
            for j, p in edges:
                if p is None and self.text[i] in scores: continue
                s = self.unknown_score if p is None else scores.get(p, NEG)
                if s != NEG: alpha[j] = _logadd(alpha[j], alpha[i] + s)
        if alpha[n] == NEG: return NEG, Counter()
        beta = [NEG] * (n + 1); beta[n] = 0.0
        for i in range(n - 1, -1, -1):
            for j, p in self.edges[i]:
                if p is None and self.text[i] in scores: continue
                s = self.unknown_score if p is None else scores.get(p, NEG)
                if s != NEG: beta[i] = _logadd(beta[i], s + beta[j])
        counts = Counter()
        for i, edges in enumerate(self.edges):
            for j, p in edges:
                if p is None and self.text[i] in scores: continue
                s = self.unknown_score if p is None else scores.get(p, NEG)
                if s != NEG and alpha[i] != NEG and beta[j] != NEG:
                    counts[p] += weight * math.exp(min(0.0, alpha[i] + s + beta[j] - alpha[n]))
        return alpha[n], counts

    def best(self, scores: dict[str, float], excluded: str | None = None):
        n = len(self.text); value = [NEG] * (n + 1); value[0] = 0.0
        previous: list[Any] = [None] * (n + 1)
        for i, edges in enumerate(self.edges):
            if value[i] == NEG: continue
            for j, p in edges:
                if p is not None and p == excluded: continue
                if p is None and self.text[i] in scores and self.text[i] != excluded: continue
                s = self.unknown_score if p is None else scores.get(p, NEG)
                candidate = value[i] + s
                if candidate > value[j]:
                    value[j] = candidate; previous[j] = (i, p)
        path = []; pos = n
        while pos and previous[pos] is not None:
            start, piece = previous[pos]; path.append(piece); pos = start
        if pos: return NEG, []
        return value[n], list(reversed(path))


def _trie(pieces: Iterable[str]) -> dict:
    root = {}
    for p in sorted(pieces):
        node = root
        for c in p: node = node.setdefault(c, {})
        node[None] = p
    return root


def prepare_records(native: pb.ModelProto, records: Iterable[dict],
                    balance_language_source: bool = True) -> tuple[list[Record], dict]:
    """Record keys: text (raw), language, source, weight (nonnegative frequency)."""
    proc = spm.SentencePieceProcessor(model_proto=native.SerializeToString())
    tags = [p.piece for p in native.pieces if p.type == USER_DEFINED]
    grouped = Counter(); examples = {}; raw_count = 0; mass = Counter()
    for r in records:
        if r.get('normalized', False):
            raise ValueError('Input must be raw text; normalized=True risks double normalization')
        text = str(r['text']); lang = str(r['language']); source = str(r['source'])
        weight = float(r.get('weight', 1.0))
        if not math.isfinite(weight) or weight < 0: raise ValueError('Invalid record weight')
        if not weight: continue
        if any(tag in text for tag in tags):
            raise ValueError('USER_DEFINED control strings require a separate parity-aware lattice')
        normalized = proc.normalize(text)
        if any(tag in normalized for tag in tags):
            raise ValueError('Normalization produced a USER_DEFINED control string; a separate parity-aware lattice is required')
        if not normalized: continue
        key = (lang, source, normalized); grouped[key] += weight
        examples.setdefault(key, text); mass[(lang, source)] += weight; raw_count += 1
    sources = defaultdict(set)
    for lang, source in mass: sources[lang].add(source)
    result = []
    for (lang, source, text), weight in sorted(grouped.items()):
        if balance_language_source:
            weight /= len(sources) * len(sources[lang]) * mass[(lang, source)]
        result.append(Record(text, weight, lang, source, examples[(lang, source, text)]))
    if not result: raise ValueError('No nonempty weighted records')
    manifest = {'raw_nonempty_rows':raw_count,'unique_language_source_texts':len(result),
                'balance_language_source':balance_language_source,
                'weighting':'Equal language mass, equal available source mass per language, supplied frequency within source' if balance_language_source else 'Caller-provided weights preserved',
                'normalized_record_sha256':_sha('\n'.join(f'{r.language}\t{r.source}\t{r.weight:.17g}\t{r.text}' for r in result).encode()),
                'per_language_mass':dict(Counter({l:sum(r.weight for r in result if r.language==l) for l in sources}))}
    return result, manifest


def _allocate(counts: dict[str, float], mass: float, floors: dict[str, float],
              ceiling: float) -> dict[str, float]:
    """Solve the bounded count-log-mass objective, then round scores to float32.

    Counts may be rescaled by a common factor without changing the optimum.
    Endpoint tolerance covers floating-point summation, not a fixed absolute
    amount of probability mass, which could hide tiny infeasible requests.
    """
    keys = sorted(counts)
    if not math.isfinite(mass) or mass < 0 or not math.isfinite(ceiling) or ceiling <= 0:
        raise ValueError('Finite nonnegative mass and positive finite ceiling required')
    if set(floors) != set(keys) or any(not math.isfinite(floors[p]) or not 0 < floors[p] <= ceiling for p in keys):
        raise ValueError('Each candidate requires a positive finite floor within the ceiling')
    if any(not math.isfinite(counts[p]) or counts[p] < 0 for p in keys):
        raise ValueError('Candidate counts must be finite and nonnegative')
    lower = sum(floors[p] for p in keys); upper = len(keys) * ceiling
    lower_tolerance = 8 * max(math.ulp(mass), math.ulp(lower))
    upper_tolerance = 8 * max(math.ulp(mass), math.ulp(upper))
    if mass < lower - lower_tolerance or mass > upper + upper_tolerance:
        raise ValueError(f'Infeasible appended mass {mass}; feasible interval [{lower}, {upper}]')
    if not keys:
        if mass: raise ValueError('Positive mass with no appended candidates')
        return {}
    # Clamp only rounding-distance endpoint differences, preventing an endless
    # multiplier search when a rounded request lies just below its floor sum.
    mass = max(lower, min(upper, mass))
    tolerance = max(8 * math.ulp(mass), mass * 1e-12)
    # A zero-usage piece gets only its lower bound. If all counts vanish, retain
    # deterministic equal residual mass rather than divide by zero.
    c = {p:float(counts[p]) for p in keys}
    if not any(c.values()): c = {p:1.0 for p in keys}
    scale = max(c.values())
    c = {p:value / scale for p, value in c.items()}
    def values(lam): return {p:max(floors[p], min(ceiling, c[p] / lam)) for p in keys}
    lo, hi = 1e-300, 1.0
    while sum(values(hi).values()) > mass: hi *= 2
    for _ in range(120):
        mid = (lo + hi) / 2
        if sum(values(mid).values()) > mass: lo = mid
        else: hi = mid
    q = values(hi)
    residue = mass - sum(q.values())
    # Zero-count bounded allocation can leave residual only at a degenerate
    # endpoint; deterministic distribution is explicit, never mass inflation.
    if residue > tolerance:
        for p in keys:
            delta = min(residue, ceiling-q[p]); q[p] += delta; residue -= delta
            if residue <= tolerance: break
    return {p:_float32(math.log(max(q[p], floors[p]))) for p in keys}


class NativeFitter:
    def __init__(self, native: bytes | pb.ModelProto, records: Iterable[dict],
                 candidates: Iterable[dict], budgets: dict[str,int],
                 protected: dict[str, Iterable[str]], *,
                 fixed_additions: dict[str,float] | None = None,
                 balance_language_source: bool = True, witness_margin: float = .001):
        self.native = load_native(native)
        self.native_bytes = self.native.SerializeToString()
        self.native_count = len(self.native.pieces)
        self.base = {p.piece:float(p.score) for p in self.native.pieces if p.type==NORMAL}
        self.native_all = {p.piece for p in self.native.pieces}
        self.low = min(self.base.values()); self.high = max(self.base.values())
        self.floor = math.exp(self.low); self.ceiling = math.exp(self.high)
        self.unknown_score = self.low - 10.0
        self.margin = witness_margin
        self.fixed_additions = dict(fixed_additions or {})
        build_model(self.native, self.fixed_additions)
        self.fixed = self.base | self.fixed_additions
        self.records, self.manifest = prepare_records(self.native,records,balance_language_source)
        self.budgets = dict(budgets)
        if not self.budgets or any(v<=0 for v in self.budgets.values()): raise ValueError('Positive group budgets required')
        self.protected = {g:set(protected.get(g,())) for g in budgets}
        self.members = defaultdict(set); self.prior = {}; self.provenance = defaultdict(list)
        for row in candidates:
            p = row['piece']; groups = set(row['groups'])
            if not p or p=='▁' or not groups or not groups <= budgets.keys():
                raise ValueError(f'Invalid candidate/group membership: {row}')
            if p in self.native_all and p not in self.base:
                raise ValueError(f'Non-NORMAL native special cannot be a fitted candidate: {p!r}')
            self.members[p].update(groups)
            score = float(row['score'])
            if not math.isfinite(score): raise ValueError('Nonfinite candidate score')
            score = max(self.low,min(self.high,score))
            self.prior[p] = max(self.prior.get(p,self.low),score)
            self.provenance[p].extend(row.get('provenance',[]))
        for g, pieces in self.protected.items():
            if len(pieces)>budgets[g]: raise ValueError(f'Mandatory pieces exceed {g} budget')
            for p in pieces:
                if not p or p=='▁': raise ValueError('Boundary is globally native, not a budgeted candidate')
                self.members[p].add(g); self.prior.setdefault(p,self.fixed.get(p,self.low))
        self.protected_union = set().union(*self.protected.values())
        self.trie = _trie(self.fixed.keys() | self.members.keys())
        self.lattices = [Lattice(r.text,self.trie,self.unknown_score) for r in self.records]
        self.literal_lattices = {p:Lattice(p,self.trie,self.unknown_score) for p in self.members}
        # Candidate match frequency is independent of current segmentation scores.
        self.occurrence = Counter()
        for r,lattice in zip(self.records,self.lattices):
            for edges in lattice.edges:
                for _,p in edges:
                    if p in self.members: self.occurrence[p]+=r.weight
        self.intake_excluded = []
        for p in list(self.members):
            if not self.occurrence[p] and p not in self.protected_union:
                self.intake_excluded.append({'piece':p,'reason':'no normalized training occurrence'})
                del self.members[p]
        for g,b in budgets.items():
            n=sum(g in groups for groups in self.members.values())
            if n<b: raise ValueError(f'{g}: only {n} evidenced/protected candidates for budget {b}')
        self.initial_members = {p:set(gs) for p,gs in self.members.items()}
        self.native_required_floors = {}
        self.native_witness_bounds = []
        for p in sorted(self.protected_union-self.fixed.keys()):
            alt,path=self.literal_lattices[p].best(self.fixed,excluded=p)
            floor=max(self.low,alt+self.margin)
            self.native_required_floors[p]=math.exp(floor)
            self.native_witness_bounds.append({'piece':p,'native_alternative_score':alt,
               'native_alternative':path,'minimum_score':floor,
               'possible_with_preserved_extrema':floor<=self.high})
        self.required_mass_floor=sum(self.native_required_floors.values())

    def policy(self) -> dict:
        new=set(self.members)-self.fixed.keys()
        return {'native_sha256':_sha(self.native_bytes),'native_count':self.native_count,
                'score_extrema':[self.low,self.high],
                'initial_candidate_added_score_mass':sum(math.exp(self.prior[p]) for p in new),
                'native_only_mandatory_minimum_mass':self.required_mass_floor,
                'mandatory_witness_bounds':self.native_witness_bounds,
                'note':'Caller chooses added_score_mass explicitly. Mandatory lower bound ignores other added competitors and may increase during joint witness projection. No globally normalized probability claim.',
                'input_manifest':self.manifest,'intake_excluded':self.intake_excluded}

    def _expectation(self,scores):
        combined=self.fixed|scores; counts=Counter(); objective=0.; per_language=defaultdict(float)
        for r,lattice in zip(self.records,self.lattices):
            value,c=lattice.expected(combined,r.weight)
            if value==NEG: raise ValueError('Unreachable lattice after removing a coverage piece')
            objective+=r.weight*value; per_language[r.language]+=r.weight*value; counts.update(c)
        return objective,counts,dict(per_language)

    def _usage(self,scores,*,details=False):
        model=build_model(self.native,self.fixed_additions|scores)
        proc=spm.SentencePieceProcessor(model_proto=model.SerializeToString())
        # Raw scorer avoids re-normalization of the already native-normalized text.
        raw=pb.ModelProto();raw.CopyFrom(model);raw.normalizer_spec.Clear()
        raw.normalizer_spec.name='identity';raw.normalizer_spec.add_dummy_prefix=False
        raw.normalizer_spec.remove_extra_whitespaces=False;raw.normalizer_spec.escape_whitespaces=False
        rawproc=spm.SentencePieceProcessor(model_proto=raw.SerializeToString())
        strings=[p.piece for p in model.pieces];counts=Counter();bylang=defaultdict(Counter)
        for start in range(0,len(self.records),256):
            batch=self.records[start:start+256]
            sequences=rawproc.encode([r.text for r in batch],out_type=int,num_threads=2)
            for r,ids in zip(batch,sequences):
                for i in ids:
                    p=strings[i];counts[p]+=r.weight;bylang[r.language][p]+=r.weight
        witnesses={}; failures=[]
        for p in sorted((set(scores)&self.protected_union)):
            body=p.removeprefix('▁')
            contexts=[body,' '+body,body+' ','a '+body+' b','a'+body+'a','अ'+body+'अ']
            idx=proc.piece_to_id(p)
            seqs=proc.encode(contexts,out_type=int,num_threads=2)
            matched=next(({'input':t,'pieces':[strings[i] for i in ids]} for t,ids in zip(contexts,seqs) if idx in ids),None)
            if matched: witnesses[p]=matched
            else: failures.append(p)
        summary={'weighted_tokens':sum(counts.values()),'weighted_unknown_tokens':counts[proc.id_to_piece(proc.unk_id())],
                 'used_new_pieces':sum(counts[p]>0 for p in scores),
                 'unused_optional_pieces':sorted(p for p in scores if p not in self.protected_union and counts[p]==0),
                 'mandatory_encoding_witnesses':witnesses,'mandatory_witness_failures':failures,
                 'per_language':{l:{'weighted_tokens':sum(c.values()),'weighted_unknown_tokens':c[proc.id_to_piece(proc.unk_id())],
                                    'used_new_pieces':sum(c[p]>0 for p in scores)} for l,c in bylang.items()}}
        if details: summary['weighted_piece_counts']=dict(counts)
        return counts,summary

    def _floors(self,scores):
        floors={p:self.floor for p in scores}
        floors.update({p:q for p,q in self.native_required_floors.items() if p in scores})
        return floors

    def _witness_project(self,scores,counts,mass,max_passes=30):
        floors=self._floors(scores); history=[]
        for iteration in range(max_passes):
            violations=[]; combined=self.fixed|scores
            for p in sorted(scores.keys()&self.protected_union):
                alternative,path=self.literal_lattices[p].best(combined,excluded=p)
                need=alternative+self.margin
                if need>self.high:
                    raise ValueError(f'Mandatory {p!r} cannot beat fixed/added alternative within native score extrema')
                if scores[p]<need-1e-6:
                    floors[p]=max(floors[p],math.exp(need))
                    violations.append({'piece':p,'required_score':need,'alternative':path})
            if not violations: return scores,history
            if sum(floors.values())>mass*(1+1e-9):
                raise ValueError(f'Mandatory witness projection requires mass >= {sum(floors.values()):.12g}, caller supplied {mass:.12g}')
            scores=_allocate({p:counts.get(p,math.exp(scores[p])) for p in scores},mass,floors,self.ceiling)
            history.append({'iteration':iteration,'raised_floors':violations,'minimum_required_mass':sum(floors.values())})
        raise ValueError('Mandatory reachability constraints did not converge at the supplied score mass')

    def _em(self,scores,mass,passes):
        trace=[]
        before,counts,bylang=self._expectation(scores)
        for iteration in range(passes):
            updated=_allocate({p:counts[p] for p in scores},mass,self._floors(scores),self.ceiling)
            after,next_counts,next_bylang=self._expectation(updated)
            if after<before-max(1e-7,abs(before)*1e-8):
                raise RuntimeError(f'Anchored EM objective decreased {before} -> {after}')
            updated,projection=self._witness_project(updated,counts,mass)
            if projection:
                projected,next_counts,next_bylang=self._expectation(updated)
            else: projected=after
            trace.append({'iteration':iteration,'objective_before':before,'objective_after_em':after,
                          'objective_after_mandatory_projection':projected,
                          'per_language_weighted_log_partition_before':bylang,
                          'maximum_score_change':max(abs(scores[p]-updated[p]) for p in scores),
                          'actual_added_mass_after_float32':sum(math.exp(s) for s in updated.values()),
                          'mandatory_projection':projection})
            scores=updated
            before,counts,bylang=projected,next_counts,next_bylang
        return scores,trace

    def _dominance(self,scores):
        result={};combined=self.fixed|scores
        for p in scores:
            alt,path=self.literal_lattices[p].best(combined,excluded=p)
            if alt>scores[p]+1e-5 and all(x is not None for x in path):
                result[p]={'alternative':path,'margin':alt-scores[p]}
        return result

    def fit(self,*,added_score_mass:float,em_passes:int=3,prune_fraction:float=.12,
            maximum_rounds:int=30,refill_rounds:int=4) -> dict:
        """Fit selected memberships to exact budgets, preserving mandatory tokens.

        Pruning loss is fused weighted Viterbi usage times local replacement gap,
        an approximation. Actual fused partition changes are logged per batch.
        Optional pieces must have deterministic TRAINING usage, not just positive
        EM expectation. A failed exact-budget/usage gate is returned explicitly;
        no arbitrary dormant fillers are inserted to make a count look correct.
        """
        if not math.isfinite(added_score_mass) or added_score_mass<=0: raise ValueError('Positive explicit added_score_mass required')
        if not 0<prune_fraction<1: raise ValueError('prune_fraction must be between 0 and 1')
        members={p:set(gs) for p,gs in self.initial_members.items()}
        new=set(members)-self.fixed.keys()
        scores=_allocate({p:math.exp(self.prior[p]) for p in new},added_score_mass,
                         self._floors(dict.fromkeys(new,self.low)),self.ceiling)
        scores,initial_projection=self._witness_project(scores,{p:math.exp(s) for p,s in scores.items()},added_score_mass)
        trace=[];removed=[]
        for round_index in range(maximum_rounds):
            scores,em=self._em(scores,added_score_mass,em_passes)
            counts,usage=self._usage(scores)
            extra={g:sum(g in gs for gs in members.values())-b for g,b in self.budgets.items()}
            if all(x<=0 for x in extra.values()):
                trace.append({'round':round_index,'em':em,'group_excess':extra,'usage':usage,'pruned':[]});break
            combined=self.fixed|scores; ranking=[]; physical_losses={}
            for p,groups in members.items():
                if p in self.fixed: loss=0.; gap=0.
                else:
                    alt,_=self.literal_lattices[p].best(combined,excluded=p)
                    gap=max(0.,scores[p]-alt);loss=counts[p]*gap
                physical_losses[p]=loss
                # Native-overlap membership costs nothing to remove: its native
                # row remains in the actual model. A shared selected piece also
                # has zero physical deletion cost until its last membership.
                for g in groups:
                    if p not in self.protected[g] and extra[g]>0:
                        physical_loss=0. if len(groups)>1 else loss
                        ranking.append((physical_loss,counts[p],len(p),p,g,gap))
            cap={g:min(n,max(1,math.ceil(prune_fraction*(self.budgets[g]+n)))) if n>0 else 0 for g,n in extra.items()}
            batch=[]
            heapq.heapify(ranking)
            while ranking:
                loss,c,length,p,g,gap=heapq.heappop(ranking)
                if cap[g]<=0 or g not in members.get(p,set()): continue
                current_loss=0. if len(members[p])>1 else physical_losses[p]
                if current_loss>loss+1e-15:
                    # A previous removal can turn shared membership into the
                    # last physical owner. Re-rank, never delete it for free.
                    heapq.heappush(ranking,(current_loss,c,length,p,g,gap));continue
                members[p].remove(g); cap[g]-=1
                if not members[p]: del members[p]
                batch.append({'piece':p,'group':g,'approximate_weighted_loss':loss,'weighted_viterbi_usage':c,'replacement_gap':gap})
            if not batch: raise ValueError('No removable optional memberships but budgets exceeded')
            before,_,_=self._expectation(scores)
            scores={p:s for p,s in scores.items() if p in members}
            after_delete,_,_=self._expectation(scores)
            scores=_allocate({p:math.exp(s) for p,s in scores.items()},added_score_mass,self._floors(scores),self.ceiling)
            scores,projection=self._witness_project(scores,{p:math.exp(s) for p,s in scores.items()},added_score_mass)
            removed.extend(batch)
            trace.append({'round':round_index,'em':em,'group_excess':extra,'usage':usage,'pruned':batch,
                          'objective_before_deletion':before,'objective_after_deletion':after_delete,
                          'post_prune_mandatory_projection':projection})
        # An evidence-based reservoir refill can replace dead entries after the
        # joint fit. Activation floors beat their current literal decomposition;
        # fixed total mass makes this a visible constrained redistribution.
        refill_trace=[]; banned=set()
        for cycle in range(refill_rounds):
            counts,usage=self._usage(scores)
            dead={p for p in scores if p not in self.protected_union and counts[p]==0}
            if not dead: break
            for p in dead:
                banned.add(p); members.pop(p,None); scores.pop(p,None)
            deficits={g:b-sum(g in gs for gs in members.values()) for g,b in self.budgets.items()}
            ranking=[]; combined=self.fixed|scores
            for p,groups in self.initial_members.items():
                if p in members or p in banned or p in self.fixed or not self.occurrence[p]: continue
                alt,path=self.literal_lattices[p].best(combined,excluded=p)
                activation=max(self.prior[p],alt+self.margin,self.low)
                if activation>self.high: continue
                # Priority uses actual observed occurrence and decomposition
                # length saved. It is not a random donor order.
                utility=self.occurrence[p]*max(0,len(path)-1)
                ranking.append((-utility,-self.occurrence[p],math.exp(activation),p,activation))
            activated={}; chosen=[]
            for negutility,negocc,m,p,activation in sorted(ranking):
                groups={g for g in self.initial_members[p] if deficits[g]>0}
                if not groups: continue
                projected_floor=self._floors(scores|activated)
                projected_floor.update({q:math.exp(s) for q,s in activated.items()})
                projected_floor[p]=math.exp(activation)
                if sum(projected_floor.values())>added_score_mass: continue
                members[p]=groups;activated[p]=activation
                for g in groups: deficits[g]-=1
                chosen.append({'piece':p,'groups':sorted(groups),'occurrence_weight':-negocc,
                               'local_token_saving_utility':-negutility,'activation_score':activation})
                if not any(deficits.values()): break
            scores.update(activated)
            if not scores: raise ValueError('No surviving appended pieces')
            floors=self._floors(scores);floors.update({p:math.exp(s) for p,s in activated.items()})
            scores=_allocate({p:math.exp(s) for p,s in scores.items()},added_score_mass,floors,self.ceiling)
            scores,projection=self._witness_project(scores,{p:math.exp(s) for p,s in scores.items()},added_score_mass)
            scores,em=self._em(scores,added_score_mass,em_passes)
            refill_trace.append({'cycle':cycle,'removed_zero_use_optional':sorted(dead),'activated':chosen,
                                 'remaining_group_deficits':deficits,'mandatory_projection':projection,'em':em})
            if any(deficits.values()): break
        counts,usage=self._usage(scores,details=True);dominance=self._dominance(scores)
        final_counts={g:sum(g in gs for gs in members.values()) for g in self.budgets}
        missing={g:sorted(self.protected[g]-{p for p,gs in members.items() if g in gs}) for g in self.budgets}
        gates={'exact_group_budgets':final_counts==self.budgets,'all_protected_selected':not any(missing.values()),
               'mandatory_encoding_witnesses':not usage['mandatory_witness_failures'],
               'all_optional_additions_used_in_training':not usage['unused_optional_pieces'],
               'no_strictly_dominated_additions':not dominance,
               'appended_mass_preserved':abs(sum(math.exp(s) for s in scores.values())-added_score_mass)<=max(1e-9,added_score_mass*1e-5)}
        model=build_model(self.native,self.fixed_additions|scores)
        selected=[{'piece':p,'score':self.fixed.get(p,scores.get(p)), 'groups':sorted(gs),
                   'native_overlap':p in self.native_all,'fixed_addition_overlap':p in self.fixed_additions,
                   'mandatory_in':[g for g in sorted(gs) if p in self.protected[g]],
                   'weighted_training_usage':counts[p],'weighted_occurrence':self.occurrence[p],
                   'provenance':self.provenance[p]} for p,gs in sorted(members.items())]
        return {'model_proto':model,'additions':scores,'selected_pieces':selected,
                'report':{'status':'passed' if all(gates.values()) else 'failed_gates',
                'gates':gates,'policy':self.policy(),'added_score_mass_requested':added_score_mass,
                'added_score_mass_actual':sum(math.exp(s) for s in scores.values()),
                'initial_mandatory_projection':initial_projection,'trace':trace,'refill_trace':refill_trace,
                'budgets':self.budgets,'final_group_counts':final_counts,'missing_protected':missing,
                'selected_memberships':sum(map(len,members.values())),
                'selected_unique_strings':len(members),'selected_native_overlap':sum(p in self.native_all for p in members),
                'new_fitted_strings':len(scores),'fixed_additions':len(self.fixed_additions),
                'final_native_vocabulary':len(model.pieces),'model_sha256':_sha(model.SerializeToString()),
                'usage':usage,'strict_dominance':dominance,
                'objective_definition':'Weighted sum of log partition over segmentations using fixed native energies and fitted appended energies constrained to a caller-selected fixed exponential mass. EM fitting is monotonic before mandatory witness projection. Pruning/refill are deterministic heuristics, not a global optimum or a normalized LM likelihood.',
                'limitations':['Training encoder usage is not a held-out efficiency or ASR accuracy result.','Script-span fitting needs independent full-record evaluation with the actual native normalizer.','Literal-context witnesses are finite evidence; unknown runs and USER_DEFINED control inputs require their own parity tests.']}}


def fit(native,records,candidates,budgets,protected,*,added_score_mass,**kwargs):
    """Convenience API. Use NativeFitter.policy() first to select an explicit mass."""
    fit_keys={'em_passes','prune_fraction','maximum_rounds','refill_rounds'}
    fit_args={k:kwargs.pop(k) for k in list(kwargs) if k in fit_keys}
    return NativeFitter(native,records,candidates,budgets,protected,**kwargs).fit(
        added_score_mass=added_score_mass,**fit_args)
