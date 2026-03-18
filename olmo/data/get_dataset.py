import inspect
import itertools

from olmo.data.academic_image_datasets import Vqa2, TextVqa, InfoQa, DocQa, AI2D, PlotQa, FigureQa, \
    DvQa, OkVqa, MMMU, CoSyn, CoSynPoint, PixmoMulitDocQa, MathVista, RealWorldQa, ChartQa, AOkVqa, \
    CountBenchQa, SceneTextQa, TabWMPDirectAnswer, TallyQa, PointBench, ScienceQAImageOnly, \
    ScreenSpotV2, OSWorldG, ScreenSpotPro
from olmo.data.academic_multi_image_datasets import MuirBench, BLINK, MMIU, MantisInstruct
from olmo.data.academic_video_datasets import (
    Tomato, TemporalBenchQa, MotionBench,
    MotionBenchCaption, MVBench, VideoEvalProMC, MLVU, LongVideoBench, LVBench, VideoMME,
    AcademicVideoPoint, EgoSchema, ActivityNet, COIN, EpicKitchens, MomentsInTime, Kinetics710,
    Youcook2, PerceptionTest, Ego4d, Ego4dCachedClips, QVHighlights, LLaVAVideoAcademic,
    CameraBenchTrain, CLEVRER, FunQA, How2QA, IntentQA, \
    SocialIQ2, SUTDTrafficQA, STAR, SportsQA, RoadTextVQA, \
    VideoLocalizedNarratives, VideoLocalizedNarrativesCaptionHf, CinepileHf, NewsVideoQA, Countix,
    Paxion, TGIF,
    TVQA, NeXTQA, CharadesSTA
)
from olmo.data.academic_video_track_datasets import (
    Mevis, MevisChallenge, Burst, ReasonVOS, RefYoutubeVOS, RefDavis17,
    LVVIS, YTVIS, ViCaS, ReVOS, MoCA,
    LVOSv1, LVOSv2, LaSOT, UWCOT, WebUOT, LaTOT, TNL2K, TNLLT,
    WebUAV, GOT10k, VastTrack, TrackingNet
)
from olmo.data.dataset import Dataset
from olmo.data.molmo2_datasets import (
    Molmo2CaptionsEval, Molmo2SynCaptionsQA, Molmo2SynCaptionsSubtitleQA, Molmo2HumanQA,
    Molmo2VideoPoint, Molmo2VideoPointEval, Molmo2VideoCountEval, Molmo2SyntheticPoint,
)
from olmo.data.molmo2_video_track_datasets import Molmo2VideoTrackInstruction, Molmo2VideoTrackEval, \
    MolmoPointTrackAny, MolmoPointTrackSyn
from olmo.data.molmo_hardcode import Molmo2HardCodes
from olmo.data.pixmo_datasets import PixMoMultiPoints, PixMoCapQa, PixMoCount, PixMoCap, \
    PixMoAskModelAnything, PixMoPoints, PixmoMultiImageQa, PixMoPointsEval
from olmo.data.text_datasets import Tulu4Filtered


# TODO Mantis

def get_dataset_by_name(dataset_name, split) -> Dataset:
    # Academic image datasets
    if dataset_name == "coco_2014_vqa_multi":
        return Vqa2(split)
    if dataset_name == "coco_2014_vqa_8192":
        return Vqa2(split, flatten_annotations=True, sample=8192)
    if dataset_name == "text_vqa":
        return TextVqa(split)
    if dataset_name == "info_qa":
        return InfoQa(split)
    if dataset_name == "doc_qa":
        return DocQa(split)
    if dataset_name == "ai2_diagram_v2_mix_transparent":
        return AI2D(split=split, boxes="both")
    if dataset_name == "plot_qa":
        return PlotQa(split)
    if dataset_name == "figure_qa":
        return FigureQa(dict(train="train", validation="validation1")[split])
    if dataset_name == "dv_qa":
        return DvQa(split)
    if dataset_name == "okvqa":
        return OkVqa(split, flatten_annotations=True)
    if dataset_name == "a_okvqa_da":
        return AOkVqa(split=split, direct_answer=True)
    if dataset_name == "a_okvqa_mc":
        return AOkVqa(split=split, direct_answer=False)
    if dataset_name in ["mmmu"]:
        return MMMU(split)
    if dataset_name in ["mmmu_test"]:
        return MMMU(split)
    if dataset_name in ["mmmu_test_v2"]:
        return MMMU(split, use_multi_image=True)
    elif dataset_name == "math_vista_v2":
        if split == "validation":
            split = "testmini"
        return MathVista(split)
    if dataset_name == "chart_qa":
        return ChartQa(split, weighted=False)
    if dataset_name == "chart_qa_weighted":
        return ChartQa(split, weighted=True)
    if dataset_name == "science_qa_img":
        return ScienceQAImageOnly(split)

    if dataset_name == "real_world_qa_no_instruction":
        assert split == "test"
        return RealWorldQa("no_instruction")
    if dataset_name == "countbench_qa":
        assert split == "huggingface"
        return CountBenchQa()
    if dataset_name == "st_qa":
        return SceneTextQa(split=split)
    if dataset_name == "tabwmp_da":
        return TabWMPDirectAnswer(split=split, include_options=False)
    if dataset_name == "tally_qa":
        return TallyQa(split=split)

    if dataset_name == "tulu4_max_2304":
        return Tulu4Filtered(split=split, max_first_msg_len=2304)
    if dataset_name == "tulu4":
        return Tulu4Filtered(split=split)

    if dataset_name == "point_bench":
        assert split == "test"
        return PointBench()
    if dataset_name == "molmo2_syn_point":
        return Molmo2SyntheticPoint(split=split)
    if dataset_name.startswith("screen_spot_v2"):
        prompt = dataset_name.split("_", maxsplit=3)[-1]
        if prompt == "v2":
            prompt = "none"
        assert split == "test"
        return ScreenSpotV2(prompt=prompt)
    if dataset_name.startswith("os_worldg"):
        prompt = dataset_name.split("_", maxsplit=2)[-1]
        if prompt == "worldg":
            prompt = "none"
        assert split == "test"
        return OSWorldG(prompt=prompt)
    if dataset_name.startswith("screen_spot_pro"):
        prompt = dataset_name.split("_", maxsplit=3)[-1]
        if prompt == "pro":
            prompt = "none"
        assert split == "test"
        return ScreenSpotPro(prompt=prompt)

    # Multi-Image data
    if dataset_name == "muir_bench":
        return MuirBench(split)
    elif dataset_name == "blink":
        return BLINK(split)
    if dataset_name == "mmiu":
        assert split == "test"
        return MMIU(split, format="multiple_choice")
    if dataset_name.startswith("mantis_instruct"):
        direct_answer = "_da" in dataset_name
        multi_image_only = "_multi_only" in dataset_name
        flat = "_flat" in dataset_name
        name = dataset_name.replace("_da", "").replace("_flat", "").replace("_multi_only", "")[len("mantis_instruct_"):]
        return MantisInstruct(name, split, direct_answer=direct_answer, multi_image_only=multi_image_only, flat=flat)


    # Cosyn data
    doc_types = [
        "chart", "chemical", "circuit", "diagram",
        "document", "graphic", "math", "music",
        "nutrition", "table"
    ]
    cosyn_dataset_names = [f"cosyn_{doc_type}{suffix}" for doc_type, suffix in
                           itertools.product(doc_types, ["", "_exp"])]
    if dataset_name == "cosyn_point":
        return CoSynPoint(split=split)
    elif dataset_name in cosyn_dataset_names:
        doc_type = dataset_name.split("_")[1]
        return CoSyn(doc_type, split=split, use_exp=dataset_name.endswith("_exp"))
    elif dataset_name.startswith("cosyn_multidoc"):
        parts = dataset_name.split("_")
        if parts[-1] == "exp":
            exp = True
            doc_type = parts[-2]
        else:
            exp = False
            doc_type = parts[-1]
        return PixmoMulitDocQa(doc_type, use_exp=exp, split=split, max_images=5)

    # PixMo Data
    if dataset_name in ["pixmo_cap_with_transcripts"]:
        return PixMoCap(split, mode="transcript_and_caption")
    if dataset_name in ["pixmo_cap"]:
        return PixMoCap(split, mode="captions")
    elif dataset_name in ["pixmo_count_train"]:
        return PixMoCount(split=split, counting="both")
    elif dataset_name in ["pixmo_count_counting"]:
        return PixMoCount(split=split, counting=True)
    elif dataset_name in ["pixmo_cap_qa"]:
        return PixMoCapQa(split=split)
    elif dataset_name in ["pixmo_cap_qa_as_user_qa"]:
        return PixMoCapQa(split=split, style="user_qa")
    elif dataset_name in ["pixmo_ask_model_anything"]:
        return PixMoAskModelAnything(split=split)
    elif dataset_name in ["pixmo_points_train"]:
        return PixMoPoints(kind="basic", split=split, counting="both", max_points=60, max_total_points_per_example=60)
    elif dataset_name in ["pixmo_points_high_freq_train"]:
        return PixMoPoints(kind="high_frequency", split=split, counting="both", max_points=60, max_total_points_per_example=60)
    if dataset_name == "pixmo_multi_points":
        return PixMoMultiPoints(split=split)
    if dataset_name == "pixmo_multi_image_qa":
        return PixmoMultiImageQa(split=split)
    if dataset_name == "pixmo_multi_image_qa_multi_only_max5":
        return PixmoMultiImageQa(split=split, multi_image_only=True, max_images=5)

    # Video dataset
    if dataset_name == "molmo2_captions_eval":
        return Molmo2CaptionsEval(split=split)
    if dataset_name == "molmo2_syn_captions_qa":
        return Molmo2SynCaptionsQA(split=split)
    if dataset_name == "molmo2_syn_captions_subtitle_qa":
        return Molmo2SynCaptionsSubtitleQA(split=split)
    if dataset_name == "molmo2_human_qa":
        return Molmo2HumanQA(split=split)
    if dataset_name.startswith("vixmo_points_oversample_no_clip"):
        return Molmo2VideoPoint(
            split=split,
            max_points=60,
            max_seconds=None,
            mode=["point_count", "point"],
            multi_message_short_clips=True,
            oversample=True,
            p_multi_turn=0.2
        )

    if dataset_name.startswith("vixmo_points_oversample"):
        return Molmo2VideoPoint(
            split=split,
            max_points=60,
            max_seconds=63,
            mode=["point_count", "point"],
            multi_message_short_clips=True,
            oversample=True,
            p_multi_turn=0.2
        )
    if dataset_name.startswith("molmo2_video_point_minmax"):
        min_points = int(dataset_name.split("_")[4])
        max_points = int(dataset_name.split("_")[5])
        return Molmo2VideoPoint(
            split=split,
            min_points=min_points,
            max_points=max_points,
            mode=["point_count", "point"],
            multi_message_short_clips=True,
            use_clips_from_metadata=True,
            max_seconds=63
        )
    if dataset_name.startswith("vixmo_points_minmax"):
        min_points = int(dataset_name.split("_")[3])
        max_points = int(dataset_name.split("_")[4])
        return Molmo2VideoPoint(
            split=split,
            min_points=min_points,
            max_points=max_points,
            mode=["point_count", "point"],
            multi_message_short_clips=True,
            use_clips_from_metadata=True,
            max_seconds=63
        )
    if dataset_name in ["molmo2_video_point_eval", "vixmo_points_point_eval"]:
        return Molmo2VideoPointEval(split=split)
    if dataset_name in ["molmo2_video_count_eval", "vixmo_points_count_clip_63s"]:
        return Molmo2VideoCountEval(split=split)
    if dataset_name == "tomato":
        return Tomato(split=split)
    if dataset_name == "mvbench":
        return MVBench(split=split)
    if dataset_name in ["motionbench", "motionbench_test"]:
        return MotionBench(split=split)
    if dataset_name in ["motionbench_caption", "motionbench_train"]:
        return MotionBenchCaption(split=split)
    if dataset_name == "temporal_bench":
        return TemporalBenchQa(split=split)
    if dataset_name == "temporal_bench_mc":
        return TemporalBenchQa(split=split, format="mc")
    if dataset_name == "qv_highlights":
        return QVHighlights(split=split)
    if dataset_name == "llava_video_mc_academic":
        return LLaVAVideoAcademic(split=split, answer_type="multi_choice")
    if dataset_name == "llava_video_oe_academic":
        return LLaVAVideoAcademic(split=split, answer_type="open_ended")
    if dataset_name == "video_eval_pro_mc":
        return VideoEvalProMC(split=split)
    if dataset_name == "mlvu":
        return VideoEvalProMC(split=split)
    if dataset_name.startswith("ego_schema"):
        return EgoSchema(split=split)
    if dataset_name == "mlvu_mc":
        return MLVU(split=split, task="multiple-choice")
    if dataset_name == "mlvu_gen":
        return MLVU(split=split, task="generation")
    if dataset_name == "long_video_bench_w_subtitle":
        return LongVideoBench(split=split, with_subtitle=True)
    if dataset_name == "lvbench":
        assert split == "test"
        return LVBench()
    if dataset_name == "video_mme_w_subtitle":
        return VideoMME(split=split, duration="all", with_subtitle=True)
    if dataset_name == "activitynet_caption":
        return ActivityNet(split=split, task="captioning")
    if dataset_name == "activitynet_qa":
        return ActivityNet(split=split, task="qa")
    if dataset_name == "activitynet_all":
        return ActivityNet(split=split, task="all")
    if dataset_name == "activitynet_all_qa":
        return ActivityNet(split=split, task="all", qa_format=True)
    if dataset_name == "coin":
        return COIN(split=split)
    if dataset_name == "coin_qa":
        return COIN(split=split, qa_format=True)
    if dataset_name == "coin_all_qa":
        return COIN(split=split, task="all", qa_format=True)
    if dataset_name == "epic_kitchens":
        return EpicKitchens(split=split)
    if dataset_name == "epic_kitchens_qa":
        return EpicKitchens(split=split, qa_format=True)
    if dataset_name == "moments_in_time":
        return MomentsInTime(split=split)
    if dataset_name == "moments_in_time_qa":
        return MomentsInTime(split=split, qa_format=True)
    if dataset_name in ["academic_video_points", "academic_points_clip_63s_2fps"]:
        return AcademicVideoPoint(split=split, max_seconds=63, mode=["point_count", "point"], max_points=60, use_clips_from_metadata=True)
    if dataset_name == "molmo2_hardcodes":
        assert split == "train"
        return Molmo2HardCodes()
    if dataset_name in ["pointing_eval_v2", "pixmo_point_eval2"]:
        assert split == "test"
        return PixMoPointsEval()
    if dataset_name == "kinetics":
        return Kinetics710(split=split)
    if dataset_name == "kinetics_qa":
        return Kinetics710(split=split, qa_format=True)
    if dataset_name == "youcook2_caption_clip":
        return Youcook2(split=split, task="caption_clip")
    if dataset_name == "youcook2_caption_start_end":
        return Youcook2(split=split, task="caption_start_end")
    if dataset_name == "youcook2_all":
        return Youcook2(split=split, task="all")
    if dataset_name == "youcook2_qa":
        return Youcook2(split=split, task="caption_clip", qa_format=True)
    if dataset_name == "youcook2_all_qa":
        return Youcook2(split=split, task="all", qa_format=True)
    if dataset_name == "perception_test":
        return PerceptionTest(split=split)
    if dataset_name == "perception_test_flat":
        return PerceptionTest(split=split, flat=True)
    if dataset_name == "perception_test_max5":
        return PerceptionTest(split=split, max_per_video=5)
    if dataset_name == "ego4d_all":
        return Ego4dCachedClips(split=split, task="all")
    if dataset_name == "ego4d_mq_label_clip":
        return Ego4d(split=split, task="mq_label_clip")
    if dataset_name == "ego4d_mq_label_start_end":
        return Ego4d(split=split, task="mq_label_start_end")
    if dataset_name == "ego4d_mq_temporal_grounding":
        return Ego4d(split=split, task="mq_temporal_grounding")
    if dataset_name == "ego4d_nlq_temporal_grounding":
        return Ego4d(split=split, task="nlq_temporal_grounding")
    if dataset_name == "camerabench_qa":
        return CameraBenchTrain(split=split)
    if dataset_name == "clevrer":
        return CLEVRER(split=split)
    if dataset_name == "funqa":
        return FunQA(split=split)
    if dataset_name == "how2qa":
        return How2QA(split=split)
    if dataset_name == "intent_qa":
        return IntentQA(split=split)
    if dataset_name == "social_iq2":
        return SocialIQ2(split=split)
    if dataset_name == "sutd_trafficqa":
        return SUTDTrafficQA(split=split)
    if dataset_name == "star":
        return STAR(split=split, max_per_video=10)
    if dataset_name == "star_mc":
        return STAR(split=split, max_per_video=10, answer_type="multi_choice")
    if dataset_name == "sportsqa_oe":
        return SportsQA(split=split)
    if dataset_name == "road_text_vqa":
        return RoadTextVQA(split=split)
    if dataset_name == "video_localized_narratives":
        return VideoLocalizedNarratives(split=split)
    if dataset_name == "video_localized_narratives_caption":
        return VideoLocalizedNarrativesCaptionHf(split=split)
    if dataset_name == "cinepile":
        return CinepileHf(split=split, with_subtitle=False)
    if dataset_name == "cinepile_with_sub":
        return CinepileHf(split=split, with_subtitle=True)
    if dataset_name == "news_video_qa":
        return NewsVideoQA(split=split, filter_empty_answers=False)
    if dataset_name == "news_video_qa_filtered":
        return NewsVideoQA(split=split, filter_empty_answers=True)
    if dataset_name == "countix_oe":
        return Countix(split=split, answer_format="oe")
    if dataset_name == "countix_mc":
        return Countix(split=split, answer_format="mc")
    if dataset_name == "paxion":
        return Paxion(split=split)
    if dataset_name == "tgif":
        return TGIF(split=split)
    if dataset_name == "tvqa":
        return TVQA(split=split)
    if dataset_name == "tvqa_with_sub":
        return TVQA(split=split, with_subtitle=True)

    if dataset_name.startswith("nextqa_mc"):
        difficulty = "all" if len(dataset_name.split("_")) == 2 else dataset_name.split("_")[2]
        return NeXTQA(split=split, task="multiple-choice", difficulty=difficulty)
    if dataset_name == "charades_sta":
        return CharadesSTA(split=split)
    if dataset_name == "charades_sta_qa":
        return CharadesSTA(split=split, qa_format=True)
    if dataset_name == "charades_sta_all":
        return CharadesSTA(split=split, task="all")
    if dataset_name == "charades_sta_all_qa":
        return CharadesSTA(split=split, task="all", qa_format=True)
    #### 
    # Academic video object tracking datasets
    # mevis: track, ground, single_point_track
    if dataset_name == "mevis_track": # Uses [1,2] sampling fps by default.
        return Mevis(split=split, task="track")
    if dataset_name == "mevis_track_1fps":
        return Mevis(split=split, task="track", sampling_fps=1)
    if dataset_name == "mevis_track_2fps":
        return Mevis(split=split, task="track", sampling_fps=2)
    if dataset_name == "mevis_ground":
        return Mevis(split=split, task="ground")
    if dataset_name == "mevis_single_point_track":
        return Mevis(split=split, task="single_point_track")
    if dataset_name == "mevis_track_eval_1fps": # Used for eval
        return Mevis(split=split, task="track", sampling_fps=1)
    if dataset_name == "mevis_valid_track_eval_1fps": # Used for codalab submission
        return MevisChallenge(task="track", sampling_fps=1)
    # ref_yt_vos: track
    if dataset_name == "ref_yt_vos_track":
        return RefYoutubeVOS(split=split, task="track")
    if dataset_name == "ref_yt_vos_track_eval_1fps": # Used for eval
        return RefYoutubeVOS(split=split, task="track", sampling_fps=1)
    # ref_davis17: track
    if dataset_name == "ref_davis17_track":
        return RefDavis17(split=split, task="track")
    if dataset_name == "ref_davis17_track_eval_1fps": # Used for eval
        return RefDavis17(split=split, task="track", sampling_fps=1)
    if dataset_name == "reasonvos_track_eval_1fps":
        return ReasonVOS(split=split, task="track", sampling_fps=1)
    # burst: track, ground, single_point_track
    if dataset_name == "burst_track":
        return Burst(split=split, task="track")
    if dataset_name == "burst_ground":
        return Burst(split=split, task="ground")
    if dataset_name == "burst_single_point_track":
        return Burst(split=split, task="single_point_track")
    # lv_vis: track, ground, single_point_track
    if dataset_name == "lv_vis_track":
        return LVVIS(split=split, task="track")
    if dataset_name == "lv_vis_ground":
        return LVVIS(split=split, task="ground")
    if dataset_name == "lv_vis_single_point_track":
        return LVVIS(split=split, task="single_point_track")
    # yt_vis: track
    if dataset_name == "yt_vis_track":
        return YTVIS(split=split, task="track")
    # vicas: track, ground, single_point_track
    if dataset_name == "vicas_track":
        return ViCaS(split=split, task="track")
    if dataset_name == "vicas_ground":
        return ViCaS(split=split, task="ground")
    if dataset_name == "vicas_single_point_track":
        return ViCaS(split=split, task="single_point_track")
    # revos: track, ground, single_point_track
    if dataset_name == "revos_track":
        return ReVOS(split=split, task="track")
    if dataset_name == "revos_ground":
        return ReVOS(split=split, task="ground")
    if dataset_name == "revos_single_point_track":
        return ReVOS(split=split, task="single_point_track")
    # moca: track, ground
    if dataset_name == "moca_track":
        return MoCA(split=split, task="track")
    if dataset_name == "moca_ground":
        return MoCA(split=split, task="ground")
    
    # Single object tracking datasets
    if dataset_name == "lvosv1_single_point_track":
        return LVOSv1(split=split, task="single_point_track")
    if dataset_name == "lvosv2_single_point_track":
        return LVOSv2(split=split, task="single_point_track")
    if dataset_name == "lasot_single_point_track":
        return LaSOT(split=split, task="single_point_track")
    if dataset_name == "uwcot_single_point_track":
        return UWCOT(split=split, task="single_point_track")
    if dataset_name == "webuot_single_point_track":
        return WebUOT(split=split, task="single_point_track")
    if dataset_name == "latot_single_point_track":
        return LaTOT(split=split, task="single_point_track")
    if dataset_name == "tnl2k_single_point_track":
        return TNL2K(split=split, task="single_point_track")
    if dataset_name == "tnllt_single_point_track":
        return TNLLT(split=split, task="single_point_track")
    if dataset_name == "webuav_single_point_track":
        return WebUAV(split=split, task="single_point_track")
    if dataset_name == "got10k_single_point_track":
        return GOT10k(split=split, task="single_point_track")
    if dataset_name == "vasttrack_single_point_track":
        return VastTrack(split=split, task="single_point_track")
    if dataset_name == "trackingnet_single_point_track":
        return TrackingNet(split=split, task="single_point_track")

    # Molmo2 video track datasets (instruction-formatted, pre-computed trajectories)
    if dataset_name == "molmo2_video_track":
        return Molmo2VideoTrackInstruction(split=split, task="track")
    if dataset_name == "molmo2_video_track_1fps":
        return Molmo2VideoTrackInstruction(split=split, task="track", sampling_fps=1)
    if dataset_name == "molmo2_video_track_2fps":
        return Molmo2VideoTrackInstruction(split=split, task="track", sampling_fps=2)
    if dataset_name == "molmo2_video_track_ground":
        return Molmo2VideoTrackInstruction(split=split, task="ground")
    if dataset_name == "molmo2_video_single_point_track":
        return Molmo2VideoTrackInstruction(split=split, task="single_point_track")
    if dataset_name == "molmo2_video_single_point_track_1fps":
        return Molmo2VideoTrackInstruction(split=split, task="single_point_track", sampling_fps=1)
    if dataset_name == "molmo2_video_single_point_track_2fps":
        return Molmo2VideoTrackInstruction(split=split, task="single_point_track", sampling_fps=2)

    # MolmoPoint Track datsets
    if dataset_name == "molmo_point_track_any":
        return MolmoPointTrackAny(split=split, task="track")
    if dataset_name == "molmo_point_track_syn":
        return MolmoPointTrackSyn(split=split, task="track")

    # Molmo2 video track eval
    if dataset_name == "molmo2_video_track_eval_1fps": # all dataset; track at 1 fps
        return Molmo2VideoTrackEval(split=split, task="track", sampling_fps=1)
    if dataset_name == "molmo2_video_track_eval_animal_1fps": # animal subset;
        return Molmo2VideoTrackEval(split=split, task="track", sampling_fps=1, configs=["animal"])
    if dataset_name == "molmo2_video_track_eval_dance_1fps": # dance subset;
        return Molmo2VideoTrackEval(split=split, task="track", sampling_fps=1, configs=["dance"])
    if dataset_name == "molmo2_video_track_eval_sports_1fps": # sports subset;
        return Molmo2VideoTrackEval(split=split, task="track", sampling_fps=1, configs=["sports"])
    if dataset_name == "molmo2_video_track_eval_person_1fps": # person subset;
        return Molmo2VideoTrackEval(split=split, task="track", sampling_fps=1, configs=["person"])
    if dataset_name == "molmo2_video_track_eval_misc_1fps": # misc subset;
        return Molmo2VideoTrackEval(split=split, task="track", sampling_fps=1, configs=["misc"])
    raise NotImplementedError(f"Dataset {dataset_name} not found")


def get_all_dataset_classes():
    """Get all dataset classes"""
    return [x for x in globals().values() if (inspect.isclass(x) and issubclass(x, Dataset))]


def get_dataset_class_by_name(dataset_name):
    """Get the class of the named dataset"""
    # We need to call `download` before initialization, so we do a bit of
    # monkey-patching so we can call `get_dataset_by_name` to get the correct class without actually
    # calling the dataset __init__ method
    if dataset_name in ["lvbench", "mmiu", "point_bench", "real_world_qa_no_instruction", "pixmo_point_eval2"]:
        split = "test"
    elif dataset_name in ["countbench_qa"]:
        split = "huggingface"
    else:
        split = "train"
    dataset_classes = get_all_dataset_classes()
    originals = {cls: cls.__init__ for cls in dataset_classes}
    try:
        for cls in dataset_classes:
            cls.__init__ = lambda *a, **kw: None
        dataset = get_dataset_by_name(dataset_name, split)
        return type(dataset)
    finally:
        for cls, init in originals.items():
            cls.__init__ = init


def download_dataset_by_name(dataset_name, n_procs=8):
    """Call `download` on the named dataset"""
    get_dataset_class_by_name(dataset_name).download(n_procs=n_procs)