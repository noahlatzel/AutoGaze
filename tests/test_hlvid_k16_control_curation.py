import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.human_gaze.curate_hlvid_k16_controls import (
    elapsed_seconds, paired_video_interval, read_rows, scores,
    seed_summary, validate_answers, video_arrays, curate, scan_events,
)


class ControlCurationTests(unittest.TestCase):
    def setUp(self):
        self.rows=[{"question_id":i,"video_path":v,"category":"av","answer":"A",
                    "prediction":p,"prediction_letter":p,"is_correct":p=="A"}
                   for i,(v,p) in enumerate([( "video1","A"),("video1","A"),("video2","B")])]

    def test_paired_source_identity_and_scores(self):
        rows=validate_answers(list(reversed(self.rows)),self.rows)
        sums,counts=video_arrays(rows,["video1","video2"])
        self.assertAlmostEqual(scores(sums,counts)["question_micro_accuracy"],2/3)
        self.assertAlmostEqual(scores(sums,counts)["macro_video_accuracy"],.5)

    def test_duplicate_question_fails(self):
        with self.assertRaisesRegex(ValueError,"Duplicate"):
            validate_answers([self.rows[0],self.rows[0],self.rows[2]],self.rows)

    def test_benchmark_video_drift_fails(self):
        changed=copy.deepcopy(self.rows); changed[0]["video_path"]="different"
        with self.assertRaisesRegex(ValueError,"identity"):
            validate_answers(changed,self.rows)

    def test_scoring_flag_drift_fails(self):
        changed=copy.deepcopy(self.rows); changed[0]["is_correct"]="true"
        with self.assertRaisesRegex(ValueError,"Scoring"):
            validate_answers(changed,self.rows)

    def test_parser_drift_fails(self):
        changed=copy.deepcopy(self.rows); changed[0]["prediction_letter"]="B"
        with self.assertRaisesRegex(ValueError,"Parser"):
            validate_answers(changed,self.rows)

    def test_video_micro_and_macro_cluster_weighting_differ(self):
        result=paired_video_interval([2,0],[2,1],seed=1,iterations=1000)
        self.assertAlmostEqual(result["question_micro_accuracy"]["difference"],2/3)
        self.assertAlmostEqual(result["macro_video_accuracy"]["difference"],.5)
        self.assertEqual(result,paired_video_interval([2,0],[2,1],seed=1,iterations=1000))

    def test_paired_identical_models_zero_interval(self):
        for stat in paired_video_interval([0,0],[2,1],seed=1,iterations=1000).values():
            self.assertEqual(stat["paired_video_ci90_low"],0)
            self.assertEqual(stat["paired_video_ci90_high"],0)

    def test_seed_interval_uses_six_seeds_not_qa_count(self):
        values=np.array([.48,.49,.50,.51,.52,.53])
        stat=seed_summary(values)
        half=2.015048373*np.std(values,ddof=1)/np.sqrt(6)
        self.assertAlmostEqual(stat["training_seed_ci90_high"]-stat["mean"],half)
        shifted=seed_summary(values-.5)
        self.assertAlmostEqual(shifted["seed_sd"],stat["seed_sd"])
        with self.assertRaisesRegex(ValueError,"six"):
            seed_summary(values[:3])

    def test_immutable_output_refused_before_reading_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError,"Output exists"):
                curate(Path("absent-config"),Path(directory))

    def test_jsonl_terminal_newline(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"rows.jsonl"
            path.write_text(json.dumps(self.rows[0]))
            with self.assertRaisesRegex(ValueError,"newline"):
                read_rows(path)

    def test_allocation_time_includes_interrupted_segments(self):
        self.assertEqual(elapsed_seconds("02:48:27")+elapsed_seconds("15:36:42"),66309)
        self.assertEqual(elapsed_seconds("1-00:35:12"),88512)

    def event_fixture(self):
        answers=copy.deepcopy(self.rows)
        events=[]
        for answer in answers:
            qid=answer['question_id']; answer['qa_key']=f'qa{qid}'
            common={'attempt_id':f'attempt{qid}','compatibility_key':'identity',
                    'question_id':qid,'recorded_at':'2026-09-13T00:00:00Z'}
            events.append({**common,'state':'attempt_started'})
            raw={'availability':'complete','min':16,'max':16,'mean':16}
            recovered={'availability':'complete','mean':64}
            events.append({**common,'state':'completed_answer','answer':answer,
                           'policy_label':'fixture','video_key':f'video{qid}',
                           'timing':{'selector_seconds':.1},
                           'context':{'context_truncated':False,'expanded_context_length':49000,
                                      'expanded_visual_tokens':48000},
                           'counters':{'raw_decoder_spatial_actions_per_tile_frame':raw,
                                       'post_adaptation_valid_patches_per_tile_frame':recovered,
                                       'post_adaptation_padded_slots_per_tile_frame':recovered},
                           'decode':{'decode_usable':True,'sampled_read_failure':False,
                                     'intended_indices':[0,10],'effective_indices':[0,10],
                                     'reported_index_mismatches':[],'tail_padding_count':0}})
        return answers,events

    def test_streamed_resume_attempts_are_not_extra_qa(self):
        answers,events=self.event_fixture()
        events.append({'state':'attempt_started','compatibility_key':'identity',
                       'attempt_id':'interrupted','recorded_at':'2026-09-13T00:01:00Z'})
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'events.jsonl'
            path.write_text(''.join(json.dumps(row)+'\n' for row in events))
            metadata,summary=scan_events(path,answers,'identity')
        self.assertEqual(len(metadata),3)
        self.assertEqual(summary['unclosed_started_attempts'],1)
        self.assertEqual(metadata[0]['expanded_context_length'],49000)

    def test_completed_decode_shift_is_a_material_blocker(self):
        answers,events=self.event_fixture()
        events[1]['decode']['effective_indices']=[0,11]
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'events.jsonl'
            path.write_text(''.join(json.dumps(row)+'\n' for row in events))
            with self.assertRaisesRegex(ValueError,'decode discrepancy'):
                scan_events(path,answers,'identity')


if __name__=="__main__":
    unittest.main()
