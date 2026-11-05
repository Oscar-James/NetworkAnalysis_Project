# NetworkAnalysis_Project

The Influence of Cartel presence on the Growth and Contraction of Mexican Drug Trafficking Groups

Research question: Does the presence of rival cartels predict where cartel–municipality links form and where they disappear?

Thesis statement:
Mexican cartels don’t expand at random. They grow out of territory they already hold, and they react to who else is there. Expansion should be strongly path-dependent: links appear where the cartel is already active in the same state. Co-presence with rivals should shape both entry and exit. Cartels that avoid each other in the network should rarely enter each other’s municipalities, and should more often disappear from shared ones.

Hypotheses:

- H1 (path dependence): a new link is more likely where the cartel is already present in the same state. Already supported: about 5.2× (95% CI 3.4–7.6).
- H2 (rival deterrence): a new link is less likely where the municipality already hosts a cartel that significantly avoids it (for example, Zetas vs Sinaloa).
- H3 (displacement): an existing link is more likely to disappear the next year when a rival that avoids it is present in the same municipality.
- H4 (heterogeneity): these effects differ by cartel; some spread across states, while others stay local.

Methods:

- Bipartite municipality–cartel networks, one per year.
- Projections onto cartels with a degree-preserving null and a state-stratified null, to find which pairs significantly avoid or overlap. These scores then serve as predictors.
- Prediction over time for link formation and link disappearance, compared against baselines that use only degree and only history.
- Per-cartel breakdowns, and community structure in a noise-corrected backbone, to describe the territories.
