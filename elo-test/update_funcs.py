from __future__ import division

import jinja2
import numpy as np
from scipy.stats import binom, norm

# Handicaps for different team sizes
magic_K = [None, -0.6, 0, 0.2, -0.1]  # 1-indexed

# This should be on the scale of typical differences between skills
sigma = 1


def _tau_phi(reds, blues):
    """Probability that the red team wins a certain point"""

    assert set(reds.keys()) & set(blues.keys()) == set()
    assert 1 <= len(reds) <= 4
    assert 1 <= len(blues) <= 4

    absI = float(len(reds))
    absJ = float(len(blues))
    K_I = magic_K[len(reds)]
    K_J = magic_K[len(blues)]
    tau = sum(reds.values()) / absI - sum(blues.values()) / absJ + K_I - K_J
    phi = sigma * np.sqrt( 1 / absI + 1 / absJ )
    return tau, phi

def _pwp_tauphi(tau, phi):
    return norm.cdf( tau / phi )

def _point_win_probability(reds, blues):
    return _pwp_tauphi(*_tau_phi(reds, blues))



def fifth_prob(reds, blues, red_score, blue_score):
    """
    reds, blues should be dictionaries mapping player_id to current
    skill.
    
    Returns two floats:  delta_red, delta_blue
    """

    assert red_score >= 0 and blue_score >= 0
    assert red_score + blue_score > 0
    assert red_score != blue_score

    tau, phi = _tau_phi(reds, blues)
    E = _pwp_tauphi(tau, phi)
    observed_E = red_score / (red_score + blue_score)
    # since E \in (0, 1), target_E \in (0, 1) and so the PPF
    # will hopefully not be +/- inf  :-)
    target_E = 0.8 * E + 0.2 * observed_E

    x = norm.ppf(target_E)
    delta = (x * phi - tau) * 0.5

    return delta, -delta

def fifth_delta(reds, blues, red_score, blue_score):
    """
    reds, blues should be dictionaries mapping player_id to current
    skill.
    
    Returns two floats:  delta_red, delta_blue
    """

    assert red_score >= 0 and blue_score >= 0
    assert red_score + blue_score > 0
    assert red_score != blue_score

    tau, phi = _tau_phi(reds, blues)
    # XXX: x could be +/- inf
    x = norm.ppf(red_score / (red_score + blue_score))
    delta = (x * phi - tau) * 0.5

    delta *= 0.2

    return delta, -delta

def linear(reds, blues, red_score, blue_score):
    """
    reds, blues should be dictionaries mapping player_id to current
    skill.
    
    Returns two floats:  delta_red, delta_blue
    """

    assert red_score >= 0 and blue_score >= 0
    assert red_score + blue_score > 0
    assert red_score != blue_score

    E = _point_win_probability(reds, blues)

    delta = (red_score * (1 - E) - blue_score * E)       \
                * 0.1 * 0.1
    return delta, -delta

def binomial(reds, blues, red_score, blue_score):
    """
    reds, blues should be dictionaries mapping player_id to current
    skill.
    
    Returns two floats:  delta_red, delta_blue
    """

    assert red_score >= 0 and blue_score >= 0
    assert red_score + blue_score > 0
    assert red_score != blue_score

    E = _point_win_probability(reds, blues)

    points = red_score + blue_score

    if red_score > blue_score:
        delta = 0.5 - binom.sf(red_score - 1, points, E)
        s_red, s_blue = 1, -1
    else:
        delta = 0.5 - binom.cdf(red_score, points, E)
        s_red, s_blue = -1, 1

    m = 2 * 0.069 * delta
    return m * s_red, m * s_blue


def predict_score(reds, blues, points=10):
    """
    Predict the score of a game; returns red_score, blue_score
    """

    E = _point_win_probability(reds, blues)

    if E > 0.5:
        return points, points * (1 - E) / E
    else:
        return points * E / (1 - E), points


functions = {
    "linear": linear,
    "binomial": binomial,
    "fifth-prob": fifth_prob,
    "fifth-delta": fifth_delta
}

columns = [{"red": 10, "blue": i} for i in range(10)] + \
          [{"red": i,  "blue": 10} for i in range(9, -1, -1)]

scenarios = [
    {"title": "even 1v1", "red": {1: 0}, "blue": {2: 0}},
    {"title": "even 2v2", "red": {1: 0, 2: 0}, "blue": {3: 0, 4: 0}},
    {"title": "even 2v1", "red": {1: 0, 2: 0}, "blue": {3: 0}},
    {"title": "diff 1v1", "red": {1: 0}, "blue": {2: 0.5}},
    {"title": "diff 2v2", "red": {1: 0, 2: 0.1}, "blue": {3: -0.3, 4: -0.1}},
]

def scoredelta_html(x):
    r, b = x
    assert -b == r

    try:
        mag = int(abs(r)*1000)
    except OverflowError:
        mag = "inf"

    if mag == 0:
        return jinja2.Markup('<span class="neutral">0</span>')
    else:
        cls = 'red' if r > 0 else 'blue'
        return jinja2.Markup('<span class="%s">+%s</span>') % (cls, mag)

def main():
    loader = jinja2.FileSystemLoader(".")
    env = jinja2.Environment(loader=loader)
    env.filters["score"] = lambda x: str(int((x + 1)*1000))
    env.filters["scoredelta"] = scoredelta_html
    template = env.get_template("template.html")
    html = template.render(functions=functions, predict=predict_score,
                           columns=columns, scenarios=scenarios)
    with open("rendered.html", "w") as f:
        f.write(html)

if __name__ == "__main__":
    main()
