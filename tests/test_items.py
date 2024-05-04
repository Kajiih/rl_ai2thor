"""Tests for the items module."""

from rl_thor.envs.sim_objects import SimObjId
from rl_thor.envs.tasks.items import CandidateData, CandidateId, ItemOverlapClass, TaskItem


def test_compute_valid_assignments_with_inherited_objects():
    cand_1_id = CandidateId(SimObjId("obj_1"))
    cand_2_id = CandidateId(SimObjId("obj_2"))
    cand_3_id = CandidateId(SimObjId("obj_3"))
    candidate_ids = [cand_1_id, cand_2_id, cand_3_id]

    item_1 = TaskItem("item_1", set())
    item_1.candidates_data = {
        cand_1_id: CandidateData(cand_1_id, item_1),
        cand_2_id: CandidateData(cand_2_id, item_1),
    }
    item_2 = TaskItem("item_2", set())
    item_2.candidates_data = {
        cand_2_id: CandidateData(cand_2_id, item_2),
        cand_3_id: CandidateData(cand_3_id, item_1),
    }

    overlap_class = ItemOverlapClass([item_1, item_2], candidate_ids)
    valid_assignments = overlap_class.valid_assignments

    expected_valid_assignments = [
        {
            item_1: cand_1_id,
            item_2: cand_2_id,
        },
        {
            item_1: cand_1_id,
            item_2: cand_3_id,
        },
        {
            item_1: cand_2_id,
            item_2: cand_3_id,
        },
    ]
    assert all(valid_assignment in valid_assignments for valid_assignment in expected_valid_assignments)
    assert all(
        expected_valid_assignment in expected_valid_assignments for expected_valid_assignment in valid_assignments
    )

    main_object = cand_2_id
    cand_2_1_id = CandidateId(SimObjId("obj_2.1"))
    cand_2_2_id = CandidateId(SimObjId("obj_2.2"))
    inherited_objects = {cand_2_1_id, cand_2_2_id}
    valid_assignments = overlap_class.compute_valid_assignments_with_inherited_objects(main_object, inherited_objects)

    expected_valid_assignments = [
        {
            item_1: cand_1_id,
            item_2: cand_2_1_id,
        },
        {
            item_1: cand_1_id,
            item_2: cand_2_2_id,
        },
        {
            item_1: cand_1_id,
            item_2: cand_3_id,
        },
        {
            item_1: cand_2_1_id,
            item_2: cand_3_id,
        },
        {
            item_1: cand_2_2_id,
            item_2: cand_3_id,
        },
    ]
    assert all(valid_assignment in valid_assignments for valid_assignment in expected_valid_assignments)
    assert all(
        expected_valid_assignment in expected_valid_assignments for expected_valid_assignment in valid_assignments
    )
