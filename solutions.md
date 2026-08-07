---
numbering:
  enumerator: "28.%s"
---
(chap_solutions)=
# Solutions to exercises

This chapter collects the solutions to the coding exercises. Sections are
numbered by the chapter whose exercises they answer.

## Chapter 9: Graph neural networks

### Exercise 9.1: A finer industry partition

The exercise asks whether a narrower definition of an industry produces a more homophilous graph, and whether a graph convolutional network trained on it does any better. The NAICS subsector code is one level below the sector code, so the construction is unchanged except for the grouping key.

```python
def homophily(blk, key):                             # same-sign pairs over all pairs
    num = den = 0.0
    for _, grp in blk.groupby(key):
        pos = int((grp['R1M_Usd'] > 0).sum()); n = len(grp); neg = n - pos
        num += pos * (pos - 1) / 2 + neg * (neg - 1) / 2
        den += n * (n - 1) / 2
    return num / max(den, 1)

univ = pd.read_csv('data/mlfi_us_universe.csv')                 # id -> sector and subsector codes
panel = panel.merge(univ[['id', 'naics_subsector_code']], on='id')
for key in ['naics_sector_code', 'naics_subsector_code']:
    h = panel.groupby('date').apply(homophily, key)
    print(f'{key}: mean h = {h.mean():.4f}')
```

```
naics_sector_code: mean h = 0.6072
naics_subsector_code: mean h = 0.6351
```

The finer partition is indeed more homophilous, though the improvement is modest: 0.6351 against 0.6072. The more striking change is in the size of the graph. At the last date of the panel the 19 sectors produce 156,196 edges, while the 76 subsectors produce 34,586, a reduction of about four fifths. A subsector neighbourhood is therefore a far more plausible set of comparable peers than a sector neighbourhood, which for the largest sector runs to several hundred firms.

Retraining the two-layer graph convolutional network on each graph, with the same seed and the same eight epochs, gives the comparison the exercise asks for. The only change to the chapter's `snapshots` is that the sector index is read off an arbitrary grouping key.

```python
def snaps_by(df, key):                               # the chapter's snapshots, grouped by `key`
    out = []
    for dt, blk in df.groupby('date'):
        out.append((dt, torch.tensor(blk[features].values, dtype=torch.float32),
                    torch.tensor(blk['y_rank'].values, dtype=torch.float32),
                    torch.tensor(pd.factorize(blk[key])[0], dtype=torch.long),
                    blk['R1M_Usd'].values))
    return out

for key, label in [('naics_sector_code', 'sector'), ('naics_subsector_code', 'subsector')]:
    parts = []
    for y in range(2015, 2027):                              # the chapter's walk-forward
        tr = snaps_by(panel[panel.date < f'{y}-01-01'], key)
        te = snaps_by(panel[panel.date.dt.year == y], key)
        parts.append(ic_series(fit_gnn(lambda: DeepGCN(K, 32, 2), tr, seed=SEED, epochs=8), te))
    s = pd.concat(parts).sort_index()
    print(f'GCN on {label}: mean rank IC {s.mean():.4f}, IR {s.mean() / s.std() * 12 ** 0.5:.2f}')
```

```
GCN on sector    : mean rank IC 0.0228, IR 0.80
GCN on subsector : mean rank IC 0.0257, IR 0.92
```

The narrower graph is better on both metrics, by about a third on the average rank IC and by a comparable margin on the information ratio. The direction is what the homophily measurement predicts, and it supports the reading of Section 9.4 that the sector graph fails less because message passing is hopeless than because a sector is too coarse a definition of a peer group: averaging a firm with three hundred loosely related companies destroys the dispersion that a stock-selection signal needs, while averaging it with thirty closer ones destroys less of it.

Two cautions belong with this result. First, the improvement is of the same order as the seed dispersion of Section 9.4.2, so a single training run does not settle it; the exercise is worth repeating over several seeds before the conclusion is trusted. Second, and more important, even the better of the two figures remains below the LightGBM benchmark of 0.0346 and below the edge-free control of 0.0276. A finer partition improves the graph without changing the verdict.

### Exercise 9.2: A point-in-time correlation graph

Here the sector graph is replaced by a correlation graph: each firm is connected to the ten others whose returns were most correlated with its own over the preceding twelve months. The exercise asks for three things, and the first is the one that matters most, namely that the construction must use no information posterior to the formation date.

The discipline is enforced by slicing the return history strictly before the date at which the graph is used.

```python
wide = data_ml.pivot(index='date', columns='id', values='R1M_Usd').sort_index()
def corr_edges(t_idx, k=10):
    """Top-k correlated peers using ONLY the 12 months strictly before date t."""
    win = wide.iloc[t_idx - 12:t_idx]                  # note: up to t-1, never t
    win = win.dropna(axis=1, thresh=10)
    C = np.array(win.corr().fillna(0).values, copy=True)
    np.fill_diagonal(C, -np.inf)                       # a firm is not its own neighbour
    return win.columns.values, np.argsort(-C, axis=1)[:, :k]
```

The slice `iloc[t_idx - 12:t_idx]` is exclusive of `t_idx`, so the correlation matrix used to build the graph at date *t* is estimated on returns realised up to *t-1*. This is the point that the equity-GNN literature most often leaves unstated: a correlation graph estimated on a window that includes the label period leaks the future into the adjacency, and the resulting performance is not attainable. Xiang et al. (2022) build their graph from the trailing prices from which the labels are derived and do not discuss the overlap.

Because the neighbourhood is now a fixed-size list rather than a clique, the aggregation is a gather over indices instead of a group mean.

```python
def neighbour_mean(X, nb_idx):
    return X[nb_idx].mean(dim=1)                       # mean over the k listed neighbours
```

Training the same two-layer architecture on this graph gives the second answer.

```
correlation graph: mean rank IC 0.0375, IR 1.28
```

This is the highest figure obtained anywhere in the chapter. It exceeds both industry partitions by a wide margin, and it exceeds the LightGBM benchmark of 0.0346 and the edge-free control of 0.0276 from Section 9.4.2. The universe is not the reason: the construction retains 97.8% of the firms over the same 134 months, so the comparison is close to like for like.

It is also, on its own, worth very little, and the reason is the one Section 9.4.3 insists on. The figure comes from a single seed. Repeating the same walk-forward under five seeds, and adding the capacity-matched control in which the neighbour average is switched off, gives a different picture.

```
CorrGCN (correlation graph)    IC 0.0252 (0.0045)  IR 0.89
CorrGCN with A_t = I           IC 0.0272 (0.0034)  IR 1.07
paired difference -0.0020 (sd 0.0023), 1 of 5 seeds favour the graph, t = -1.96
```

Averaged over seeds the correlation graph attains 0.0252, not 0.0375, so the headline figure was a favourable initialisation rather than a property of the graph. Against its own matched control the graph again loses, by 0.0020 of rank IC, in four seeds out of five. That difference is not significant at conventional levels, and we would not claim the correlation graph is actively harmful; what we can say is that it does not earn its edges either.

The comparison with Section 9.4.3 is nonetheless instructive, and it is the most useful thing in this exercise. The sector graph costs 0.0060 of rank IC against its matched control, with no seed favouring it; the correlation graph costs 0.0020, with one seed favouring it. Better edges do hurt less, which is what the homophily reasoning of Section 9.3.3 predicts. But the improvement moves the penalty towards zero rather than through it, so on this panel the difficulty is not only that industry membership is a poor proxy for co-movement. Even edges estimated directly from co-movement fail to carry information the node's own characteristics do not already contain, which is a harder problem for the graph programme than a bad taxonomy would have been.

The third question concerns turnover, and it produces the most interesting number of the exercise.

```python
cur = {(cols[a], cols[b]) for a in range(len(cols)) for b in nb[a]}
turnover.append(1 - len(cur & prev) / max(len(cur), 1))   # share of edges that changed
print(f'mean monthly edge turnover: {np.mean(turnover):.1%}')
```

```
mean monthly edge turnover: 42.8%
```

Roughly two edges in five are replaced from one month to the next. The graph is therefore not a stable description of the economy but a rolling statistic with a substantial estimation error, and a model trained on it is being asked to learn from a structure that is largely different each time it is used. This has a practical consequence beyond prediction quality: if positions depend on the neighbourhood, a graph that turns over at 43% per month will contribute turnover to the portfolio itself, which is a cost the prediction metrics of Section 9.4 do not capture at all. The sector graph, whatever its shortcomings in homophily, has an edge turnover of essentially zero.

## Chapter 12: Genetic algorithms for portfolio optimisation and factor investing

### Exercise 12.1: Multi-objective optimisation with turnover

This exercise adds a third objective, turnover, to the return-versus-risk problem of §11.5.4. Turnover is measured as the $L_1$ distance between the candidate weights and the previously held weights, so the algorithm must now trade expected return against both volatility and the cost of moving the book. We use DEAP's `selNSGA2` with a three-dimensional fitness.

```python
import numpy as np, random
from deap import base, creator, tools, algorithms

w_prev = np.full(len(mu), 1 / len(mu))                        # last period's weights (equal-weight start)
creator.create('Tri', base.Fitness, weights=(1.0, -1.0, -1.0))   # max return, min vol, min turnover
creator.create('Port', list, fitness=creator.Tri)
def evaluate(ind):
    w = np.clip(ind, 0, None); w = w / (w.sum() + 1e-12)
    ret = w @ mu; vol = np.sqrt(w @ cov @ w + 1e-12)
    turn = np.abs(w - w_prev).sum()                           # L1 turnover versus previous book
    return ret, vol, turn
```

The registration mirrors §11.5.4 but swaps in `selNSGA2` and evolves the population with `eaMuPlusLambda`, after which the first Pareto front is plotted as a three-dimensional scatter.

```python
tb = base.Toolbox(); tb.register('w', random.random)
tb.register('ind', tools.initRepeat, creator.Port, tb.w, len(mu)); tb.register('pop', tools.initRepeat, list, tb.ind)
tb.register('evaluate', evaluate); tb.register('mate', tools.cxSimulatedBinaryBounded, eta=15, low=0, up=1)
tb.register('mutate', tools.mutPolynomialBounded, eta=20, low=0, up=1, indpb=0.1); tb.register('select', tools.selNSGA2)
pop = tb.pop(n=200); algorithms.eaMuPlusLambda(pop, tb, 200, 200, 0.7, 0.3, 60, halloffame=None, verbose=False)
front = tools.sortNondominated(pop, len(pop), first_front_only=True)[0]
import matplotlib.pyplot as plt
F = np.array([ind.fitness.values for ind in front])
ax = plt.figure().add_subplot(projection='3d')
ax.scatter(F[:, 0], F[:, 1], F[:, 2]); ax.set_xlabel('return'); ax.set_ylabel('vol'); ax.set_zlabel('turnover')
```

The bi-objective efficient frontier of §11.5.4 becomes a curved surface once turnover enters: for any target return there is now a family of portfolios trading volatility against trading activity. The practical reading is that the low-turnover corner of the surface sacrifices only a small amount of expected return relative to the aggressive corner, which is exactly the information a cost-conscious investor needs to choose a portfolio ex post rather than committing to fixed objective weights in advance. This is the central argument of §11.5.1 for computing the whole front instead of a single scalarised optimum.

### Exercise 12.2: Parameter sensitivity heatmap

This exercise is a controlled study of the guidance in §11.3.9. We rerun the feature-selection GA of §11.4.1 over a grid of population sizes, mutation rates, and generation counts, recording the out-of-sample IC of the selected subset for each configuration, and we summarise the result as a heatmap.

```python
import numpy as np, pandas as pd, seaborn as sns, matplotlib.pyplot as plt

def run_ga(pop_size, mut_rate, ngen):                        # returns OOS IC of the selected subset
    # ... identical to the feature-selection GA of §11.4.1, but with tb.mutate indpb=mut_rate and eaSimple(..., ngen=ngen)
    mask = fit_feature_ga(pop_size=pop_size, mut_rate=mut_rate, ngen=ngen)
    cols = [features[j] for j in np.flatnonzero(mask)]
    return oos(cols)[0]

records = []
for ps in [20, 50, 100, 200]:
    for mr in [0.01, 0.02, 0.05, 0.10]:
        records.append((ps, mr, run_ga(ps, mr, ngen=50)))    # hold generations at 50 for the grid
grid = pd.DataFrame(records, columns=['pop', 'mut', 'ic']).pivot(index='pop', columns='mut', values='ic')
sns.heatmap(grid, annot=True, fmt='.3f', cmap='viridis'); plt.title('OOS IC by population and mutation rate')
```

The heatmap reproduces the qualitative message of §11.3.9: performance is a plateau rather than a sharp peak, and it is broadly insensitive to the exact settings within the sensible range. Populations of 50 to 100 combined with mutation rates near $1/L$ to $2/L$ (here, with $L=122$, roughly 0.01 to 0.02) sit comfortably on the plateau, while very small populations occasionally underperform through premature convergence. Increasing the generation count buys little once the population has converged and, if the fitness is a noisy proxy for out-of-sample IC, extra generations can even erode results by fitting the validation folds more tightly, the same overfitting-through-search mechanism documented in §11.6.5.

### Exercise 12.3: Weighted factor composites

This exercise generalises the binary feature-selection chromosome of §11.4.1 to a hybrid encoding: for each of the 122 features we carry a binary inclusion gene and a real-valued weight gene. The included, weighted features are combined into a single composite score whose IC is the fitness, so the GA now decides both which factors to use and how much to lean on each.

```python
import numpy as np, random
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler

K = len(features)
Z_tr = StandardScaler().fit_transform(training_sample[features])   # standardise so weights are comparable
Z_te = StandardScaler().fit_transform(testing_sample[features])

def decode(ind):                                             # first K genes = inclusion bits, next K = weights
    b = np.array(ind[:K]) > 0.5; w = np.array(ind[K:]) * b
    return w / (np.abs(w).sum() + 1e-12)                     # normalised weight vector

def composite_ic(ind, Z, y):
    return spearmanr(Z @ decode(ind), y).correlation
```

The hybrid genome uses arithmetic (blend) crossover on the real weights and bit-flip-style mutation on the inclusion bits; the two halves are recombined and evaluated jointly, then the best composite is compared out of sample against an equal-weighted version of the same selected subset and against the short list.

```python
def blend_flip(a, b, alpha=0.5):                             # arithmetic crossover on weights, uniform swap on bits
    for i in range(K, 2 * K):                                # weights: BLX-style blend
        g = alpha * a[i] + (1 - alpha) * b[i]; a[i], b[i] = g, alpha * b[i] + (1 - alpha) * a[i]
    for i in range(K):                                       # bits: uniform crossover
        if random.random() < 0.5: a[i], b[i] = b[i], a[i]
    return a, b

# ... standard DEAP loop with evaluate = composite_ic on the training standardised matrix, seed fixed ...
best = evolve_hybrid(seed=7)                                 # returns best individual
sel = [features[j] for j in np.flatnonzero(np.array(best[:K]) > 0.5)]
ic_w = composite_ic(best, Z_te, testing_sample['R1M_Usd'].values)
ic_eq = spearmanr(testing_sample[sel].mean(axis=1), testing_sample['R1M_Usd']).correlation
ic_sl = spearmanr(testing_sample[features_short].mean(axis=1), testing_sample['R1M_Usd']).correlation
print(f'weighted composite IC = {ic_w:+.4f}')
print(f'equal-weight subset IC = {ic_eq:+.4f}')
print(f'short-list IC          = {ic_sl:+.4f}')
```

The weighted composite usually edges out the equal-weighted version of the same subset in sample, because the extra 122 continuous weights give the GA more degrees of freedom to fit the training IC. Out of sample, however, that advantage typically shrinks or disappears: the additional parameters are additional opportunities to overfit, and an equal weighting of a well-chosen subset is a strong, low-variance benchmark. This mirrors the central tension of the chapter, that the GA is an effective optimiser whose flexibility must be disciplined by out-of-sample validation, and it echoes the §11.6.3 finding that the parsimonious short list remains difficult to beat once genuine out-of-sample evaluation replaces in-sample fit.

## Chapter 23: Classical NLP for factor investing

### Exercise 23.1: Loughran-McDonald sector tone

The exercise has four steps: (i) ingest the LM word lists, (ii) preprocess each filing into tokens, (iii) score each filing as the net tone (positive minus negative tokens, normalised by length), and (iv) aggregate scores by industry sector. We illustrate the core of the pipeline; the SEC EDGAR pulls and sector merge are assumed to have been performed beforehand and stored in a dataframe with columns `cik`, `ticker`, `sector`, `filing_text`.

```python
import pandas as pd, re, nltk
nltk.download('punkt_tab', quiet=True)
lm = pd.read_csv('Loughran-McDonald_MasterDictionary.csv')       # from authors' website
pos = set(lm.loc[lm['Positive'] > 0, 'Word'].str.lower())
neg = set(lm.loc[lm['Negative'] > 0, 'Word'].str.lower())

def net_tone(text):
    tokens = [t for t in re.findall(r'[a-zA-Z]+', text.lower())]
    if not tokens: return 0.0
    p = sum(1 for t in tokens if t in pos)
    n = sum(1 for t in tokens if t in neg)
    return (p - n) / len(tokens)                                  # normalise by length

filings = pd.read_parquet('edgar_10k_excerpts.parquet')           # cik, sector, filing_text
filings['lm_tone'] = filings['filing_text'].apply(net_tone)
sector_stats = filings.groupby('sector')['lm_tone'].agg(['mean', 'std', 'count'])
print(sector_stats.sort_values('mean'))
```

The expected pattern in published work (Loughran and McDonald, 2011; Bushee, Gow, and Taylor, 2018) is a systematic ordering: financials and energy sit at the negative end (heavy use of LM words like *loss*, *impair*, *default*, *litigation*, *restate*), regulated utilities sit near the middle, and consumer-discretionary and technology sit at the less-negative end (more forward-looking language and product-narrative prose). A reader who fits a one-way ANOVA on `lm_tone ~ sector` should obtain an F-statistic well above the 5% critical value, confirming that the sector effect is significant rather than a sampling artifact. The right interpretation is *not* that financials are gloomier firms but that the LM dictionary inherits the conservative, risk-disclosing vocabulary that accounting and legal teams in those sectors are required to use.

### Exercise 23.2: LDA topics and next-month returns

We split this into the topic-modelling pass and the predictive-regression pass.

```python
import pandas as pd, gensim, re
from gensim import corpora
from gensim.models import LdaModel
news = pd.read_csv('financial_headlines.csv')                     # date, ticker, headline
docs = [re.findall(r'[a-zA-Z]+', h.lower()) for h in news['headline']]
dictionary = corpora.Dictionary(docs)
dictionary.filter_extremes(no_below=5, no_above=0.5)              # drop ultra-rare and very common
corpus = [dictionary.doc2bow(d) for d in docs]
lda = LdaModel(corpus, num_topics=10, id2word=dictionary,
               passes=10, random_state=42)
for k, words in lda.show_topics(num_topics=10, num_words=10, formatted=False):
    print(k, [w for w, _ in words])
```

A typical 10-topic LDA on financial headlines surfaces interpretable themes: earnings & guidance, M&A activity, macro / Fed policy, energy & commodities, regulation & litigation, tech & product launches, dividend & buyback announcements, downgrades & analyst actions, geopolitics, and a residual catch-all topic. We then convert each headline into a topic-proportion vector and aggregate to a monthly firm-level feature panel.

```python
import numpy as np, statsmodels.api as sm
topic_vecs = np.array([[p for _, p in lda.get_document_topics(b, minimum_probability=0)]
                       for b in corpus])
news_topics = pd.concat([news[['date', 'ticker']],
                         pd.DataFrame(topic_vecs, columns=[f'topic_{i}' for i in range(10)])],
                        axis=1)
news_topics['month'] = pd.to_datetime(news_topics['date']).dt.to_period('M')
panel = news_topics.groupby(['ticker', 'month']).mean(numeric_only=True).reset_index()

returns = pd.read_csv('next_month_returns.csv')                    # ticker, month, ret_next
df = panel.merge(returns, on=['ticker', 'month'])
X = sm.add_constant(df[[f'topic_{i}' for i in range(10)]])
fit = sm.OLS(df['ret_next'], X).fit(cov_type='HAC', cov_kwds={'maxlags': 3})
print(fit.summary().tables[1])
```

In a Newey-West-adjusted regression of next-month returns on the ten topic proportions, most coefficients will not reach conventional significance, in line with the broader finding of the chapter that text features carry weak average signals at the single-name monthly horizon. Topics that *do* tend to load are the earnings / guidance topic (positive, consistent with the post-earnings-announcement drift; see Chan, 2003) and the downgrades / litigation topic (negative, consistent with Tetlock, 2007). A reader who repeats the exercise on a longer history and a wider topic count will recover the qualitative result of Nguyen and Shirai (2015) that combining topic structure with sentiment scoring outperforms either signal alone.

## Chapter 24: Word and sentence embeddings

### Exercise 24.1: Word2Vec on financial text

The exercise trains a Word2Vec model on a corpus of financial documents and inspects the resulting embedding space via nearest-neighbour queries.

```python
import pandas as pd, re, gensim
from gensim.models import Word2Vec
filings = pd.read_parquet('edgar_10k_excerpts.parquet')          # cik, sector, filing_text
docs = [re.findall(r'[a-zA-Z]+', t.lower()) for t in filings['filing_text']]
model = Word2Vec(sentences=docs, vector_size=200, window=5,
                 min_count=10, workers=4, epochs=15, seed=42)    # skip-gram-style training
for term in ['growth', 'risk', 'dividend', 'acquisition']:
    print(term, '->', [w for w, _ in model.wv.most_similar(term, topn=8)])
```

The expected pattern on a financial corpus of meaningful size (a few hundred 10-Ks or several thousand news headlines) is domain-coherent neighbours: `growth → expansion, increase, revenues, sales, momentum`; `risk → exposure, uncertainty, volatility, liability, default`; `dividend → distribution, payout, declared, shareholders, repurchase`; `acquisition → purchase, merger, divestiture, transaction, integration`. Two readings are useful. First, the neighbours of `risk` are strongly accounting-flavoured (`liability`, `default`) rather than statistically flavoured (`variance`, `quantile`), confirming the Loughran-McDonald (2011) point that financial corpora encode the legal sense of common words. Second, polysemy is visible: `growth` neighbours mix biological-sounding words (`expansion`) with strictly financial ones (`revenues`), which a single static embedding cannot disentangle. This last failure mode is the structural limitation that Chapter 22 addresses by moving to contextual embeddings.

### Exercise 24.2: Sentence-transformer peer similarity and NAICS sectors

We encode the business-description section of each 10-K with `all-MiniLM-L6-v2`, compute pairwise cosine similarity, and visualise how peer structure relates to NAICS sectors.

```python
import pandas as pd, numpy as np
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
filings = pd.read_parquet('edgar_business_descriptions.parquet')  # ticker, naics_sector, body
model = SentenceTransformer('all-MiniLM-L6-v2')                   # 384-dim sentence encoder
embeddings = model.encode(filings['body'].tolist(),
                          batch_size=16, show_progress_bar=False,
                          normalize_embeddings=True)               # unit-norm => dot product = cos
sim = embeddings @ embeddings.T                                    # pairwise cosine similarity
order = filings.sort_values('naics_sector').index.to_numpy()
plt.imshow(sim[order][:, order], cmap='viridis'); plt.colorbar()
plt.title('Pairwise cosine similarity, firms ordered by NAICS sector')
plt.show()
```

The heatmap reorganised by NAICS sector typically shows visible block structure: within-sector cosine similarity is markedly higher than cross-sector similarity, with the cleanest blocks for sectors whose firms have homogeneous business descriptions (utilities, real estate, finance). Three diagnostics are worth running on the matrix. (i) **Within-vs-cross-sector means.** The within-sector average cosine similarity is typically 0.55 to 0.70, while the cross-sector average is 0.30 to 0.45, a 0.15 to 0.25 gap that confirms the embedding captures industry structure. (ii) **Misclassified peers.** The off-diagonal entries with the highest cosine similarity often reveal text-based peers that the sector code does not group together (an industrial conglomerate paired with a defence prime; an apparel brand paired with a luxury house). These are the embedding analogues of the Hoberg and Phillips (2016) text-based peer industries. (iii) **Diffuse blocks.** Broad, heterogeneous sectors such as manufacturing and information produce more diffuse blocks than utilities or finance, reflecting the genuine breadth of those NAICS buckets and signalling that text similarity is a strict refinement of the standard classification in those regions of the universe.

### Exercise 24.3: Sentiment-versus-characteristics SHAP horse race

The exercise reproduces the within-model comparison of Section 21.7.4 on a reduced feature set. We take a quarterly slice of the numerical panel, merge the latest earnings-call sentiment onto each stock-quarter with no look-ahead, target the per-quarter percentile rank of the forward return, and read pooled SHAP values off a single gradient-boosted fit. The interest is the *relative* standing of the sentiment scores against the characteristics, not predictive accuracy.

```python
import pandas as pd, numpy as np
dml = pd.read_parquet('mlfi_us_data.parquet')                        # numerical panel
q = dml[dml['date'].dt.month.isin([3, 6, 9, 12])].copy()           # quarter-ends
chars = ['PB', 'PE', 'ROE', 'mom_252', 'vol_252', 'size_log']      # a handful of characteristics
q = q[['id', 'date', 'R3M_Usd'] + chars].sort_values('date')

sent = pd.read_parquet('sentiment_panel.parquet')                  # earnings-call scores
sent = sent[sent['secid'].notna()].rename(columns={'secid': 'id'})
sent['id'] = sent['id'].astype(int)
score_cols = ['finbert_prep_mgmt', 'lm_tone_prep_mgmt', 'vader_prep_mgmt']
sent = sent[['id', 'date'] + score_cols].sort_values('date')
```

We merge the latest call at or before each quarter-end, target the cross-sectional percentile rank of the forward return, and rank the features by mean absolute SHAP value.

```python
bundle = pd.merge_asof(q, sent, on='date', by='id',                # latest call <= quarter-end
                       direction='backward', tolerance=pd.Timedelta('185D'))
bundle = bundle.dropna(subset=['R3M_Usd', 'finbert_prep_mgmt'])
bundle['y'] = bundle.groupby('date')['R3M_Usd'].rank(pct=True)     # per-quarter percentile target
feats = chars + score_cols

import xgboost as xgb, shap
fit = xgb.XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.02,
                       subsample=0.8, colsample_bytree=0.7, random_state=42)
fit.fit(bundle[feats], bundle['y'])
sv = shap.TreeExplainer(fit).shap_values(bundle[feats])
rank = pd.Series(np.abs(sv).mean(axis=0), index=feats).sort_values(ascending=False)
print(rank.round(4))
```

The strongest sentiment feature, the Loughran-McDonald tone on prepared remarks, lands fifth of the nine, in the middle of the ranking: above the earnings-based ratios (price-to-earnings and return-on-equity) but below the price-trend, size, and volatility characteristics. This reproduces on a small scale the mid-tier standing that the full 154-feature horse race of Section 21.7.4 documents, where the strongest sentiment score reaches only eighth of the full set.

### Exercise 24.4: Sentiment-regime stratification of a long-short factor

We first build the market-level sentiment state, the per-quarter cross-sectional mean of the three-segment FinBERT score, and cut it into terciles. We then form a long-short decile sort on the per-stock sentiment signal each quarter and split its returns by the prevailing regime, annualising a Sharpe within each. The comparison of interest is directional: the classical result predicts the long-short is strongest following high sentiment.

```python
import pandas as pd, numpy as np
sent = pd.read_parquet('sentiment_panel.parquet')
fb = ['finbert_prep_mgmt', 'finbert_qa_mgmt', 'finbert_qa_analyst']
sent = sent[sent['secid'].notna()].copy()
sent['s_mean'] = sent[fb].mean(axis=1, skipna=True)                # per-call sentiment
sent['quarter'] = sent['date'].dt.to_period('Q').dt.to_timestamp('Q')
s_mkt = sent.groupby('quarter')['s_mean'].mean()                   # aggregate composite
regime = pd.qcut(s_mkt, 3, labels=['Low', 'Med', 'High'])         # sentiment regime
```

We form the top-minus-bottom decile portfolio on the signal each quarter, using the return to the next call as the holding-period return, and compute an annualised Sharpe within each regime.

```python
sq = (sent.groupby(['secid', 'quarter'], as_index=False)
          .agg(sig=('s_mean', 'mean'), r=('fwd_to_next_call', 'mean')).dropna())

def long_short(g):                                                 # top minus bottom decile
    d = pd.qcut(g['sig'], 10, labels=False, duplicates='drop')
    return g.loc[d == d.max(), 'r'].mean() - g.loc[d == d.min(), 'r'].mean()
ls = sq.groupby('quarter').apply(long_short).rename('ls')
tab = pd.concat([ls, regime.rename('regime')], axis=1).dropna()
sharpe = tab.groupby('regime')['ls'].agg(lambda x: np.sqrt(4) * x.mean() / x.std())
print(sharpe.reindex(['Low', 'Med', 'High']).round(2))
```

The long-short Sharpe is positive in the low and medium regimes (0.32 and 0.60) but negative in the high regime (-0.21): the factor does not strengthen following high sentiment, it weakens, the contrarian pattern of TABLE 21.4. This runs against the Stambaugh, Yu, and Yuan (2012) prediction that anomaly long-short premia are largest after high sentiment, a discrepancy the chapter reads as the sentiment signal being largely priced by the time aggregate sentiment is elevated.

## Chapter 25: Transformers and pre-trained language models for finance

### Exercise 25.1: FinBERT [CLS] similarity of the six sentences

We load FinBERT as a plain encoder (rather than through the sentiment pipeline of Section 22.7), take the `[CLS]` token vector as a fixed-length summary of each sentence, unit-normalise the vectors so that a dot product is a cosine, and read off the pairwise similarities.

```python
import numpy as np, torch
from transformers import AutoTokenizer, AutoModel
sample_texts = [                                             # the six sentences of Section 22.7
    "Revenue growth was strong driven by innovation across our beverage portfolio.",
    "The softness in price mix was attributed to unfavorable category mix and constrained production capacity.",
    "Operating losses did not materialize, and margins held firm.",
    "Free cash flow improved and management sees favorable conditions for the remainder of the year.",
    "Adverse macroeconomic conditions and persistent inflation pose a risk to near-term consumer demand.",
    "The firm achieved outstanding results with superior margin expansion in North America.",
]
tok = AutoTokenizer.from_pretrained('ProsusAI/finbert')
model = AutoModel.from_pretrained('ProsusAI/finbert'); model.eval()

embs = []
with torch.no_grad():
    for s in sample_texts:
        out = model(**tok(s, return_tensors='pt'))
        embs.append(out.last_hidden_state[0, 0].numpy())     # [CLS] token = position 0
E = np.array(embs)
E = E / np.linalg.norm(E, axis=1, keepdims=True)             # unit-norm => dot = cosine
S = E @ E.T
print(np.round(S, 2))
```

```
## [[ 1.00 -0.02  0.62  0.86 -0.08  0.91]
##  [-0.02  1.00  0.42  0.11  0.70  0.04]
##  [ 0.62  0.42  1.00  0.80  0.22  0.71]
##  [ 0.86  0.11  0.80  1.00 -0.03  0.93]
##  [-0.08  0.70  0.22 -0.03  1.00 -0.08]
##  [ 0.91  0.04  0.71  0.93 -0.08  1.00]]
```

The most similar pair the model identifies is Doc 3 and Doc 5, at a cosine of 0.93: "Free cash flow improved and management sees favorable conditions" and "The firm achieved outstanding results with superior margin expansion". Both are unambiguously upbeat, and the wider pattern is that the `[CLS]` space groups the sentences by *sentiment polarity* rather than by *topic*. The three positive sentences (Docs 0, 3, 5) form a tight cluster with mutual cosines of 0.86 to 0.93, while the two clearly negative sentences (Docs 1 and 4) are each other's nearest neighbour (cosine 0.70) and sit far from the positive cluster (cosines near zero or negative), even though Doc 0 is about revenue, Doc 3 about cash flow, and Doc 5 about margins, three different topics. Doc 2 ("Operating losses did not materialize, and margins held firm"), whose surface words are negative but whose meaning is positive, is pulled toward the positive cluster (cosine 0.71 with Doc 5), a further sign that the representation is organised around meaning rather than vocabulary. The lesson is that the `[CLS]` embedding of a sentiment-fine-tuned encoder is effectively a sentiment space, not a topical-similarity space: it answers "do these sentences share a tone?" well and "do these firms discuss the same business?" poorly. When the goal is peer or topical similarity, the contrastively-trained sentence-transformer of Section 21.4 is the appropriate tool, exactly as noted at the end of Section 22.7.

## Chapter 26: Large language models for factor investing

### Exercise 26.1: The calibration anchor and cross-sectional dispersion

The exercise has three parts: pull the text from EDGAR, score it twice (with and without the calibration anchor), and compare the dispersion of the two score sets. Because each reader's ten firms and choice of model differ, the exact numbers will vary; the qualitative result does not. We first fetch the most recent Risk Factors section for a handful of tickers. EDGAR is free and asks only for a descriptive `User-Agent` header.

```python
import requests, re
HEADERS = {'User-Agent': 'Student research (student@example.edu)'}

tickers = ['AAPL', 'MSFT', 'NVDA', 'JPM', 'XOM', 'PG', 'KO', 'PFE', 'BA', 'CAT']
cik_map = {d['ticker']: d['cik_str'] for d in                 # ticker -> zero-padded CIK
           requests.get('https://www.sec.gov/files/company_tickers.json',
                        headers=HEADERS).json().values()}

def risk_factors(ticker):                                    # most recent 10-K, Item 1A
    cik = cik_map[ticker]
    sub = requests.get(f'https://data.sec.gov/submissions/CIK{cik:010d}.json',
                       headers=HEADERS).json()['filings']['recent']
    i = sub['form'].index('10-K')
    acc = sub['accessionNumber'][i].replace('-', '')
    url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/{sub['primaryDocument'][i]}"
    text = re.sub(r'<[^>]+>', ' ', requests.get(url, headers=HEADERS).text)   # strip HTML tags
    m = re.search(r'Item\s*1A\.?\s*Risk Factors(.*?)Item\s*1B', text, re.S | re.I)
    return re.sub(r'\s+', ' ', m.group(1)).strip()[:12000] if m else ''       # ~3k tokens

texts = {t: risk_factors(t) for t in tickers}
```

We then score each disclosure on a single forward-risk-severity axis, once with the calibration anchor in the system prompt and once without it, reusing the Section 23.8 call pattern.

```python
from groq import Groq
import json, numpy as np
client = Groq()                                              # GROQ_API_KEY in the environment

ANCHOR = (" Calibrate on the population: a typical large-cap 10-K risk section scores 0.0, "
          "and reserve |score| > 0.5 for clearly above- or below-average risk.")
BASE = ("You are an equity analyst. Score the forward-risk severity of the 10-K Risk Factors "
        "text on a continuous scale from -1.0 (benign) to +1.0 (severe).{anchor} "
        'Return only JSON: {{"risk_severity": <float>}}.')

def score(text, anchored):
    system = BASE.format(anchor=ANCHOR if anchored else "")
    r = client.chat.completions.create(
        model='llama-3.1-8b-instant', temperature=0.2,
        response_format={'type': 'json_object'},
        messages=[{'role': 'system', 'content': system},
                  {'role': 'user', 'content': text}])
    return json.loads(r.choices[0].message.content)['risk_severity']

anchored   = np.array([score(t, True)  for t in texts.values()])
anchorless = np.array([score(t, False) for t in texts.values()])
print(f"cross-sectional std  anchored={anchored.std():.3f}  anchorless={anchorless.std():.3f}")
```

The pattern to expect, robust across ticker choices and models, is that the anchored scores spread the firms across the range while the anchorless scores collapse into a narrow band of mild positive severity: without an explicit zero, the model falls back on a default "risks are non-trivial but manageable" register and rates almost every large-cap filing similarly. The anchored standard deviation is accordingly several times the anchorless one, consistent with the roughly threefold dispersion gap that Section 23.8 reports on our validation sample.

Which version should feed a factor sort? The anchored one, for a mechanical rather than an aesthetic reason: a cross-sectional quintile sort ranks firms against each other within each period, so a signal whose cross-sectional standard deviation is near zero produces near-random quintiles, and the small measurement noise in the LLM's scores swamps the tiny genuine spread. The calibration anchor is precisely the instruction that forces the model to differentiate firms rather than report an absolute "how risky is this in general" reading, which is why Section 23.8 singles it out as the single most consequential line in the prompt. A natural extension is to repeat the comparison with a stronger model (for example Llama-3.3-70B): this typically narrows the anchorless gap but does not close it, because the compression is a property of the task framing rather than of model capacity.

## Chapter 27: End-to-end case study: building a text-based equity factor

### Exercise 27.1: Replicating the case study with a different model and a thinner base

We rerun the case study on the saved panel with two deliberate changes: we replace the gradient-boosted trees with a *random forest* (a bagging ensemble rather than a boosting one), and we compare the full 122-characteristic base against a thinned 20-characteristic base, each with and without the three NLP features. The forest is a genuinely different learner, so the exercise probes whether the chapter's conclusions are properties of the signal or of the specific model.

```python
import pandas as pd, numpy as np, re
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import spearmanr
panel = pd.read_parquet('case_study_panel.parquet')
panel = panel[panel['s_llm_4d'].notna() & panel['R1M_Usd'].notna()].copy()

nlp = ['s_llm_4d_r', 's_finbert_r', 's_lm_r']            # the 3 uniformized NLP features
fbase = sorted([c for c in base_characteristics          # 122 characteristics with <5% missing
                if panel[c].isna().mean() < 0.05])
thin = fbase[:20]                                         # a hand-picked 20-feature subset
for c in fbase:                                           # median-by-date imputation
    panel[c] = panel.groupby('date')[c].transform(lambda x: x.fillna(x.median()))
sep = pd.Timestamp('2014-01-15')
tr, te = panel[panel.date < sep], panel[panel.date >= sep]
rf = dict(n_estimators=200, max_depth=8, max_features='sqrt',
          min_samples_leaf=50, n_jobs=-1, random_state=42)

def fit_pred(feats):                                      # random forest, raw R1M_Usd target
    return RandomForestRegressor(**rf).fit(tr[feats], tr['R1M_Usd']).predict(te[feats])
```

We form the four predictions and read their cross-sectional information coefficients and the gross long-short Sharpe of the quintile sort, exactly as in Sections 24.10 and 24.11.

```
##                  IC       Sharpe   Turn
## full122 base   +0.0203    0.419   0.326
## full122 +NLP   +0.0173    0.407   0.334
## thin20  base   -0.0006    0.704   0.123
## thin20  +NLP   -0.0091    0.745   0.164
## full122 +filter           0.617   0.393
## thin20  +filter           0.766   0.198
```

Three observations answer the exercise. First, the marginal contribution of the NLP *features* does grow as the base is thinned, exactly as Section 24.12 predicts: adding the three features moves the gross Sharpe by $-0.012$ on the full 122-characteristic base (from 0.419 to 0.407, no help) but by $+0.042$ on the thinned base (from 0.704 to 0.745). When the base already captures most of the cross-sectional signal, an incremental text feature is redundant; when the base is deliberately starved, the text carries information the characteristics no longer supply. Second, the ranking of the two base sets flips relative to the chapter: with a random forest the thinned base *outperforms* the full one (0.704 against 0.419), the opposite of the boosted-tree result, because an unregularized forest splitting on `sqrt` of 122 mostly-collinear characteristics dilutes each tree with noisy features, whereas gradient boosting with shrinkage tolerates the wider set. This is the intended lesson: the "more features is better" finding of Section 24.11 is model-specific, not universal. Third, the sentiment *filter* helps under both bases and both models, lifting the thinned-base Sharpe from 0.704 to 0.766 and the full-base Sharpe from 0.419 to 0.617; it is the most portable of the three ways to use sentiment because it acts as a veto on the base model's extreme trades rather than as a signal the model must learn.

The filter thresholds and the transaction-cost assumption both matter, and in opposite directions.

```
## filter thresholds (thin20 base, gross Sharpe)
##   lo=0.20 / hi=0.80 :  +0.854      # mild veto: drop only the most conflicted 20%
##   lo=0.34 / hi=0.66 :  +0.766      # the chapter's default
##   lo=0.50 / hi=0.50 :  +0.622      # aggressive: veto every below/above-median name
##
## cost sensitivity (thin20 +filter)
##   gross            :  +0.766
##   10 bps/turnover  :  +0.716
##   20 bps/turnover  :  +0.666
##   40 bps/turnover  :  +0.565
```

A milder filter is better here: vetoing only the most egregiously conflicted fifth of each leg preserves breadth and lifts the gross Sharpe to 0.854, while the aggressive median cut over-thins the legs and drops it to 0.622. On costs, the thinned model's low turnover (roughly 0.20 per leg against the full model's 0.39) is what lets the filtered strategy degrade gracefully, retaining a Sharpe near 0.57 even at a punitive 40 basis points per unit of turnover; the cost figures are illustrative, since a realistic net number depends on venue, borrow, and execution as noted in Section 24.11. The overall message reproduces the chapter's on a different learner: the sentiment features earn their keep only where the base is weak, while the sentiment filter is the robust way to convert the signal into portfolio value.
