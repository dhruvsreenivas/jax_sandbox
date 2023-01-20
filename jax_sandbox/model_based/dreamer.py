import jax
import jax.numpy as jnp
import haiku as hk
import optax
import distrax
import functools
from typing import NamedTuple, Dict, Optional, Tuple

from jax_sandbox.common.nets import *
from jax_sandbox.common.dataset import TransitionBatch
from jax_sandbox.common.utils import opt_class, initializer, stack_dict

State = Dict[str, jnp.ndarray]

class DreamerState(NamedTuple):
    wm_params: hk.Params
    wm_opt_state: optax.OptState
    rng_key: jax.random.PRNGKey
    
class LayerNormGRU(hk.RNNCore):
    def __init__(self, hidden_size, norm=False, init='glorot_uniform'):
        super().__init__()
        self._size = hidden_size
        self._layer = hk.Linear(3 * hidden_size, with_bias=norm is not None, w_init=initializer(init))
        
        if norm:
            self._norm = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
        else:
            self._norm = None
            
    def initial_state(self, batch_size: Optional[int]):
        state = jnp.zeros(self._size)
        if batch_size is not None:
            state = jnp.broadcast_to(state, (batch_size,) + state.shape)
        return state
    
    def __call__(self, x, h):
        xh = jnp.concatenate([x, h], axis=-1)
        parts = self._layer(xh)
        if self._norm is not None:
            parts = self._norm(parts)
        
        reset, cand, update = jnp.split(parts, 3, -1)
        reset = jax.nn.sigmoid(reset)
        cand = jax.lax.tanh(reset * cand) # TODO: can set activation here in init if needed
        update = jax.nn.sigmoid(update - 1)
        out = update * cand + (1 - update) * h
        return out, out # have to return 2 things as par for hk.RNNCore
    
def get_gru(gru_type: str, **kwargs) -> hk.RNNCore:
    if gru_type == 'norm':
        return LayerNormGRU(**kwargs)
    else:
        del kwargs['norm']
        gru_fn = functools.partial(
            hk.GRU,
            w_i_init=kwargs['init'],
            w_h_init=hk.initializers.Orthogonal()
        )
        return gru_fn(**kwargs)
    
class RSSM(hk.Module):
    '''Recurrent state space model.'''
    def __init__(self, cfg):
        super().__init__()
        self._n_models = cfg.n_models
        self._discrete_dim = cfg.discrete_dim
        self._discrete = self._discrete_dim > 1
        
        self._deter_dim = cfg.deter_dim
        self._stoch_dim = cfg.stoch_dim
        
        w_init = initializer(cfg.init)
        act = activation(cfg.act)
        self._std_act = activation(cfg.std_act)
        self._min_std = cfg.min_std
        
        # recurrent stack
        self._pre_gru = hk.Linear(self._deter_dim, w_init=w_init)
        kwargs = {'hidden_size': self._deter_dim, 'init': 'glorot_uniform'}
        self._gru = get_gru(cfg.gru_type, **kwargs)
        
        # prior + post
        self._prior = hk.Sequential([
            hk.Linear(cfg.hidden_dim, w_init=w_init),
            act
        ])
        if self._discrete_dim > 1:
            # we are in discrete Dreamer mode
            self._prior_dist = hk.Linear(self._stoch_dim * self._discrete_dim, w_init=w_init)
        else:
            self._prior_dist = hk.Linear(self._stoch_dim * 2, w_init=w_init)
        
        self._post = hk.Sequential([
            hk.Linear(cfg.hidden_dim, w_init=w_init),
            act
        ])
        if self._discrete_dim > 1:
            self._post_dist = hk.Linear(self._stoch_dim * self._discrete_dim, w_init=w_init)
        else:
            self._post_dist = hk.Linear(self._stoch_dim * 2, w_init=w_init)
    
    def init_state(self, bs: Optional[int]) -> State:
        def create_state():
            if self._discrete:
                return {
                    'logits': jnp.zeros((bs, self._stoch_dim, self._discrete_dim)),
                    'stoch': jnp.zeros((bs, self._stoch_dim, self._discrete_dim)),
                    'deter': jnp.zeros((bs, self._deter_dim))
                }
            else:
                return {
                    'mean': jnp.zeros((bs, self._stoch_dim)),
                    'std': jnp.zeros((bs, self._stoch_dim)),
                    'stoch': jnp.zeros((bs, self._stoch_dim)),
                    'deter': jnp.zeros((bs, self._deter_dim))
                }
        
        if bs is None:
            if self._discrete:
                return {
                    'logits': jnp.zeros((self._stoch_dim, self._discrete_dim)),
                    'stoch': jnp.zeros((self._stoch_dim, self._discrete_dim)),
                    'deter': jnp.zeros((self._deter_dim))
                }
            else:
                return {
                    'mean': jnp.zeros(self._stoch_dim),
                    'std': jnp.zeros(self._stoch_dim),
                    'stoch': jnp.zeros(self._stoch_dim),
                    'deter': jnp.zeros(self._deter_dim)
                }
        else:
            return create_state()
            
    def get_dist_and_info(self, stats: jnp.ndarray) -> Tuple[distrax.Distribution, Dict[str, jnp.ndarray]]:
        if self._discrete:
            stats = jnp.reshape(stats, stats.shape[:-1] + (self._stoch_dim, self._discrete_dim))
            dist = distrax.OneHotCategorical(logits=stats)
            dist = distrax.straight_through_wrapper(dist)
            dist = distrax.Independent(dist, 1)
            info = {'logits': stats}
        else:
            mean, std = jnp.split(stats, 2, -1)
            dist = distrax.MultivariateNormalDiag(mean, std)
            dist = distrax.Independent(dist, 1)
            info = {'mean': mean, 'std': std}
        
        return dist, info
            
    def get_feature(self, state: State) -> jnp.ndarray:
        stoch = state['stoch']
        deter = state['deter']
        if self._discrete:
            stoch = jnp.reshape(stoch, stoch.shape[:-2] + (self._discrete_dim * self._stoch_dim,))
            
        return jnp.concatenate([stoch, deter], axis=-1)
    
    def onestep_prior(self,
                      embed: Optional[jnp.ndarray],
                      action: jnp.ndarray,
                      mask: jnp.ndarray,
                      state: Optional[State] = None) -> Tuple[State, jnp.ndarray]:
        # embed: (B, embed_dim), action: (B, action_dim), mask: (B, 1), state: State
        del embed
        B = action.shape[0]
        if state is None:
            state = self.init_state(B)
        
        deter, stoch = state['deter'], state['stoch']
        deter *= mask
        stoch *= mask
        
        if self._discrete:
            stoch = jnp.reshape(stoch, stoch.shape[:-2] + (self._stoch_dim * self._discrete_dim,))
        
        x = jnp.concatenate([stoch, action], axis=-1)
        x = self._pre_gru(x)
        new_deter, _ = self._gru(x, deter)
        
        prior = self._prior(new_deter)
        prior_stats = self._prior_dist(prior)
        prior_dist, stats = self.get_dist_and_info(prior_stats)
        
        new_stoch = prior_dist.sample(seed=hk.next_rng_key())
        new_state = {
            'stoch': new_stoch,
            'deter': new_deter
        }
        new_state.update(stats)
        
        new_feature = self.get_feature(new_state)
        return new_state, new_feature, None
    
    def onestep(self,
                embed: jnp.ndarray,
                action: jnp.ndarray,
                mask: jnp.ndarray,
                state: Optional[State] = None) -> Tuple[State, State, jnp.ndarray]:
        prior_state, _ = self.onestep_prior(action, mask, state)
        x = jnp.concatenate([prior_state['deter'], embed], axis=-1)
        x = self._post(x)
        post_stats = self._post_dist(x)
        post_dist, stats = self.get_dist_and_info(post_stats)
        
        new_stoch = post_dist.sample(seed=hk.next_rng_key())
        new_state = {
            'stoch': new_stoch,
            'deter': prior_state['deter']
        }
        new_state.update(stats)

        new_feature = self.get_feature(new_state)
        return new_state, new_feature, prior_state
    
    def __call__(self,
                 embed_seq: jnp.ndarray,
                 action_seq: jnp.ndarray,
                 mask_seq: jnp.ndarray,
                 state: Optional[State] = None,
                 prior: bool = False) -> Tuple[Dict, Dict, jnp.ndarray]:
        
        B = embed_seq.shape[1]
        if state is None:
            state = self.init_state(B)
        
        # TODO find out how to make this efficient using jax.lax.scan or fori_loop
        onestep_fn = self.onestep_prior if prior else self.onestep
        posts = []
        features = []
        priors = []
        for embed, action, mask in zip(embed_seq, action_seq, mask_seq):
            state, feature, extra = onestep_fn(embed, action, mask, state)
            posts.append(state)
            features.append(feature)
            if not prior:
                priors.append(extra)
        
        posts = stack_dict(posts) # {k: (T, B, D)}
        features = jnp.stack(features) # (T, B, feature_dim)
        if not prior:
            priors = stack_dict(priors)
            
        return posts, priors, features
        
class WorldModel(hk.Module):
    def __init__(self, cfg):
        super().__init__()
        act = activation(cfg.act)
        w_init = initializer(cfg.init)
        
        # encoder, rssm + decoder
        self._encoder = hk.Sequential([
            hk.Conv2D(cfg.depth, kernel_shape=4, stride=2, padding='VALID', w_init=w_init),
            act,
            hk.Conv2D(cfg.depth * 2, kernel_shape=4, stride=2, padding='VALID', w_init=w_init),
            act,
            hk.Conv2D(cfg.depth * 4, kernel_shape=4, stride=2, padding='VALID', w_init=w_init),
            act,
            hk.Conv2D(cfg.depth * 8, kernel_shape=4, stride=2, padding='VALID', w_init=w_init),
            act
        ])
        
        self._rssm = RSSM(cfg.rssm)
        
        self._pre_decoder = hk.Linear(32 * cfg.depth, w_init=w_init)
        self._decoder = hk.Sequential([
            hk.Conv2DTranspose(cfg.depth * 4, kernel_shape=5, stride=2, padding='VALID', w_init=w_init),
            jax.nn.elu,
            hk.Conv2DTranspose(cfg.depth * 2, kernel_shape=5, stride=2, padding='VALID', w_init=w_init),
            jax.nn.elu,
            hk.Conv2DTranspose(cfg.depth, kernel_shape=6, stride=2, padding='VALID', w_init=w_init),
            jax.nn.elu,
            hk.Conv2DTranspose(cfg.obs_shape[-1], kernel_shape=6, stride=2, padding='VALID', w_init=w_init),
            jax.nn.elu
        ])
        
        self._reward_head = hk.Sequential([
            hk.Linear(cfg.hidden_dim, w_init=w_init),
            jax.nn.elu,
            hk.Linear(cfg.hidden_dim, w_init=w_init),
            jax.nn.elu,
            hk.Linear(cfg.hidden_dim, w_init=w_init),
            jax.nn.elu,
            hk.Linear(cfg.hidden_dim, w_init=w_init),
            jax.nn.elu
        ])
        self._discount_head = hk.Sequential([
            hk.Linear(cfg.hidden_dim, w_init=w_init),
            jax.nn.elu,
            hk.Linear(cfg.hidden_dim, w_init=w_init),
            jax.nn.elu,
            hk.Linear(cfg.hidden_dim, w_init=w_init),
            jax.nn.elu,
            hk.Linear(cfg.hidden_dim, w_init=w_init),
            jax.nn.elu
        ])
        
    def onestep(self, action: jnp.ndarray, state: Optional[State]):
        if state is None:
            state = self._rssm.init_state(None)
        
        state, feature, _ = self._rssm.onestep_prior(None, action, jnp.ones_like(action), state)
        img_mean = self._decoder(self._pre_decoder(feature))
        reward_mean = self._reward_head(feature)
        disc_logits = self._discount_head(feature)
        
        img = self.get_img_or_reward_dist(img_mean).sample(seed=hk.next_rng_key())
        reward = self.get_img_or_reward_dist(reward_mean, is_img=False).sample(seed=hk.next_rng_key())
        disc = self.get_discount_dist(disc_logits).sample(seed=hk.next_rng_key())
        
        return state, (img, reward, disc)
        
    def get_img_or_reward_dist(self, mean: jnp.ndarray, is_img: bool = True) -> distrax.Distribution:
        dist = distrax.Normal(mean, 1.0)
        if is_img:
            return distrax.Independent(dist, 3)
        else:
            return distrax.Independent(dist, 0)
        
    def get_discount_dist(self, disc_logits: jnp.ndarray) -> distrax.Distribution:
        dist = distrax.Bernoulli(logits=disc_logits)
        return distrax.Independent(dist, 0)
        
    def __call__(self, 
                 obs_seq: jnp.ndarray,
                 action_seq: jnp.ndarray,
                 mask_seq: jnp.ndarray,
                 state: Optional[State] = None,
                 prior: bool = False):
        
        embed_seq = hk.BatchApply(self._encoder)(obs_seq) # (T, B, embed_dim)
        posts, priors, features = self._rssm(embed_seq, action_seq, mask_seq, state, prior=prior)
        
        img_feats = hk.BatchApply(self._pre_decoder)(features)
        img_mean = hk.BatchApply(self._decoder)(img_feats)
        reward_mean = hk.BatchApply(self._reward_head)(features)
        disc_logits = hk.BatchApply(self._discount_head)(features)
        
        img_dist = self.get_img_or_reward_dist(img_mean)
        reward_dist = self.get_img_or_reward_dist(reward_mean, is_img=False)
        disc_dist = self.get_discount_dist(disc_logits)
        dists = {'img': img_dist, 'reward': reward_dist, 'discount': disc_dist}

        return posts, priors, dists
    
class Dreamer:
    def __init__(self, cfg):
        # set up
        def wm():
            model = WorldModel(cfg)
            
            def observe(obs_seq: jnp.ndarray,
                        action_seq: jnp.ndarray,
                        mask_seq: jnp.ndarray,
                        state: Optional[State]):
                return model(obs_seq, action_seq, mask_seq, state, prior=False)

            def imagine(obs_seq: jnp.ndarray,
                        action_seq: jnp.ndarray,
                        mask_seq: jnp.ndarray,
                        state: Optional[State]):
                return model(obs_seq, action_seq, mask_seq, state, prior=True)
            
            def init(obs_seq: jnp.ndarray,
                        action_seq: jnp.ndarray,
                        mask_seq: jnp.ndarray,
                        state: Optional[State]):
                return model(obs_seq, action_seq, mask_seq, state, prior=False)
            
            return init, (observe, imagine)
        
        model = hk.multi_transform(wm)