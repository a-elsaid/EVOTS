# nas_ts/engine.py

from __future__ import annotations

import os
from typing import Optional
import torch
import json
from pathlib import Path

from loguru import logger
from rich.progress import (
    Progress,
    BarColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    SpinnerColumn,
    TextColumn,
)

from ..core.individual import Individual
from ..core.population import Population
from ..backends.backend_base import EvaluationBackend
from ..utils.logger import Logger
from ..utils.checkpoint import save_checkpoint
from .selection_ops import tournament_selection, find_worst
from .evolution_ops import reproduce
from .genome_init import random_genome
# from .repair import repair_genome


class EvolutionEngine:
    def __init__(
        self,
        exp_cfg,
        backend: EvaluationBackend,
        logger_: Optional[Logger] = None,
        checkpoint_dir: Optional[str] = None,
    ):
        self.exp_cfg = exp_cfg
        self.backend = backend

        pop_size = exp_cfg.evolution.population_size
        self.population = Population(pop_size)

        # Completed evaluations (what your progress bar should track)
        self.eval_count = 0

        # Total jobs ever submitted (completed + in-flight)  critical for threaded backend
        self.total_submitted = 0

        # indiv_id -> Individual (in-flight only)
        self.submitted = {}

        # Best individual seen so far
        self.best_individual: Optional[Individual] = None

        self.logger_obj = logger_
        self.checkpoint_dir = checkpoint_dir

        self.evo_config = exp_cfg.evolution

        # Track best-so-far (single objective assumed: lower fitness is better)
        self.best_fitness = None
        self.best_eval_id = None
        self.best_genome_id = None

        # Prefer saving inside the run folder you're already using:
        # logs_dir/run_name/...
        if self.checkpoint_dir is None and self.logger_obj is not None:
            self.checkpoint_dir = str(Path(self.logger_obj.log_dir) / "checkpoints")

        if self.checkpoint_dir is not None:
            Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------
    # Helper to save best individual seen so far
    # ------------------------------------------
    def _maybe_save_best(self, indiv) -> None:

        best_genome_path = (
            Path(self.exp_cfg.run_info.logs_dir)
            / self.exp_cfg.run_info.name
            / f"{self.exp_cfg.run_info.name}__best.pt"
        )

        # Only meaningful for single-objective fitness
        if indiv.fitness is None:
            return

        # Need the returned weights from the evaluator
        metrics = indiv.metrics or {}
        state = metrics.get("state_dict_cpu", None)
        if state is None:
            return

        is_better = (self.best_fitness is None) or (indiv.fitness < self.best_fitness)
        if not is_better:
            return

        self.best_fitness = float(indiv.fitness)
        self.best_eval_id = int(metrics.get("eval_id", -1))
        self.best_genome_id = getattr(indiv, "id", None)

        # Ensure directory exists for best checkpoint
        best_genome_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "eval_id": self.best_eval_id,
            "generation": int(metrics.get("generation", -1)),
            "fitness": self.best_fitness,
            "metrics": {k: v for k, v in metrics.items() if not k.startswith("_")},
            "meta": metrics.get("meta", None),
            "model_state_dict": state,
            "genome": indiv.genome.to_dict() if hasattr(indiv, "genome") else None,
        }

        torch.save(payload, best_genome_path)

        # human-readable marker
        with open(best_genome_path.with_suffix(".json"), "w") as f:
            json.dump(
                {
                    "eval_id": self.best_eval_id,
                    "generation": payload["generation"],
                    "fitness": self.best_fitness,
                    "genome_id": self.best_genome_id,
                },
                f,
                indent=2,
            )
        indiv.genome.dump_structure(best_genome_path.with_suffix(".genome.json"))


    # -------------------------
    # Init population (submits initial jobs)
    # -------------------------
    def initialize(self):
        ss = self.exp_cfg.search_space
        evo = self.exp_cfg.evolution
        constraints = self.exp_cfg.genome_constraints

        pop_size = evo.population_size
        seed_fraction = evo.init_seed_fraction
        seeds = self.exp_cfg.seed_genomes or []

        # Helper: submit job if we still have budget
        def _submit(ind: Individual) -> bool:
            if self.total_submitted >= evo.max_evals:
                return False
            logger.info(f"[Init] Submitting indiv {ind.id}")
            self.backend.submit(ind.id, ind.genome)
            self.submitted[ind.id] = ind
            self.total_submitted += 1
            return True

        # 1) Seeded individuals
        if seeds and seed_fraction > 0.0:
            max_seed = int(pop_size * seed_fraction)
            num_seed = min(len(seeds), max_seed)

            for i in range(num_seed):
                if self.total_submitted >= evo.max_evals:
                    return
                # ind = Individual(repair_genome(genome=seeds[i], ss=ss))
                ind = Individual(genome=seeds[i])
                logger.info(f"[Init] Seeded individual: \n{ind.genome.to_structure_str()}")
                _submit(ind)

        # 2) Fill remaining slots with random genomes (up to pop_size, but also <= max_evals)
        while (self.population.size() + len(self.submitted)) < pop_size:
            if self.total_submitted >= evo.max_evals:
                return
            g = random_genome(ss, constraints)
            ind = Individual(genome=g)
            logger.info(f"[Init] Seeded individual:: \n{ind.genome.to_structure_str()}")
            _submit(ind)

    # -------------------------
    # One engine tick: poll completed jobs and spawn children (within budget)
    # -------------------------
    def step(self) -> int:
        sel_cfg = self.exp_cfg.selection_config
        ss = self.exp_cfg.search_space
        evo = self.exp_cfg.evolution
        constraints = self.exp_cfg.genome_constraints

        completed = self.backend.poll()  # list of (id, metrics)
        num_completed = 0

        for indiv_id, metrics in completed:
            # Completed job must exist in submitted
            indiv = self.submitted.pop(indiv_id, None)
            
            # if indiv is None:
                # logger.warning(f"[Engine] Received completion for unknown indiv_id {indiv_id}. Ignoring.")
                # ...
                

            # Attach eval_id + generation fields
            eval_id = self.eval_count
            metrics = metrics or {}
            metrics["eval_id"] = eval_id
            gen = eval_id // self.evo_config.evals_per_generation
            metrics["generation"] = gen

            indiv.metrics = metrics

            # indiv.metrics = metrics
            # if "state_dict_cpu" not in metrics:
            #     logger.warning(f"[Engine] Missing state_dict_cpu for indiv {indiv_id}. Best checkpoint will NOT be saved.")

            # Fitness for single-objective
            if sel_cfg.mode == "single":
                indiv.fitness = sel_cfg.aggregation(metrics)

            # Track best individual (genome only; no weights needed)
            if indiv.fitness is not None:
                if (self.best_fitness is None) or (indiv.fitness < self.best_fitness):
                    self.best_fitness = float(indiv.fitness)
                    self.best_individual = indiv

            # Insert newcomer (steady-state)
            if self.population.size() < self.population.max_size:
                self.population.add(indiv)
            else:
                worst = find_worst(self.population, sel_cfg)
                if sel_cfg.mode == "single":
                    if indiv.fitness is not None and worst.fitness is not None:
                        if indiv.fitness < worst.fitness and worst is not self.best_individual:
                            self.population.remove(worst)
                            self.population.add(indiv)
                else:
                    self.population.add(indiv)
                    worst = find_worst(self.population, sel_cfg)
                    if worst is not self.best_individual: # don't remove best individual
                        self.population.remove(worst)

            # Log to disk
            if self.logger_obj is not None:
                self.logger_obj.log_evaluation(indiv)

            # Maybe save best-so-far checkpoint
            self._maybe_save_best(indiv)

            # Console log
            if indiv.fitness is not None:
                logger.info(f"[Eval {self.eval_count+1}] GenomeID={indiv.genome.id} fitness={indiv.fitness:.4f}")
            else:
                logger.info(f"[Eval {self.eval_count+1}] GenomeID={indiv.genome.id} (no fitness)")

            self.eval_count += 1
            num_completed += 1

            # Spawn exactly one child per completion IF we still have submission budget
            if self.total_submitted < evo.max_evals:
                if self.population.size() > 0:
                    parent1 = tournament_selection(self.population, sel_cfg, evo.tournament_size)
                    if self.population.size() > 1:
                        parent2 = tournament_selection(self.population, sel_cfg, evo.tournament_size)
                    else:
                        parent2 = parent1

                    child_genome = reproduce(parent1.genome, parent2.genome, ss, evo, constraints)
                    child = Individual(genome=child_genome)

                    logger.info(f"[Spawn] New child {child.genome.id} Created with Genome: \n{child_genome.to_structure_str()}")

                    self.backend.submit(child.id, child.genome)
                    self.submitted[child.id] = child
                    self.total_submitted += 1

                    logger.info(f"[Spawn] New child {child.genome.id} submitted. (submitted={self.total_submitted}/{evo.max_evals})")

            if self.checkpoint_dir is not None and self.eval_count > 0 and (self.eval_count % 50 == 0):
                os.makedirs(self.checkpoint_dir, exist_ok=True)
                weight_pool = getattr(self.backend, "weight_pool", None)
                ckpt_path = os.path.join(self.checkpoint_dir, f"ckpt_{self.eval_count}.pt")
                save_checkpoint(ckpt_path, self.population, self.eval_count, weight_pool)

                gen = self.eval_count // self.evo_config.evals_per_generation
                if self.logger_obj is not None:
                    self.logger_obj.log_generation(self.population, gen)

        return num_completed

    # -------------------------
    # Main loop
    # -------------------------
    def run(self):
        self.initialize()
        evo = self.exp_cfg.evolution

        last_eval_count = self.eval_count

        with Progress(
            SpinnerColumn(),
            TextColumn("[bold green]{task.description}"),
            BarColumn(),
            TextColumn("[cyan]{task.completed}/{task.total}"),
            TextColumn("• Gen [yellow]{task.fields[generation]}"),
            TextColumn("• Best [magenta]{task.fields[best]:.4f}"),
            TextColumn("• Pop [blue]{task.fields[pop]}"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            refresh_per_second=5,
        ) as progress:

            task_id = progress.add_task(
                "Evaluating",
                total=evo.max_evals,
                generation=0,
                best=float("inf"),
                pop=0,
            )

            # Run until we have COMPLETED max_evals (not just submitted)
            while self.eval_count < evo.max_evals:
                self.step()

                new_evals = self.eval_count - last_eval_count
                if new_evals > 0:
                    gen = self.eval_count // self.evo_config.evals_per_generation

                    if self.population.size() > 0:
                        fits = [ind.fitness for ind in self.population.get_all() if ind.fitness is not None]
                        best = min(fits) if fits else float("inf")
                    else:
                        best = float("inf")

                    progress.update(
                        task_id,
                        advance=new_evals,
                        generation=gen,
                        best=best,
                        pop=self.population.size(),
                    )
                    last_eval_count = self.eval_count

            # Final generation log
            if self.logger_obj is not None:
                gen = self.eval_count // self.evo_config.evals_per_generation
                self.logger_obj.log_generation(self.population, gen)

            # Final checkpoint
            if self.checkpoint_dir is not None:
                os.makedirs(self.checkpoint_dir, exist_ok=True)
                weight_pool = getattr(self.backend, "weight_pool", None)
                ckpt_path = os.path.join(self.checkpoint_dir, f"ckpt_final_{self.eval_count}.pt")
                save_checkpoint(ckpt_path, self.population, self.eval_count, weight_pool)

        return self.population, self.best_individual