from .enums import *
from .welford import Welford


def generate_snapshot_creation(vtk_output: bool, target: Target, grid: Grid, field_map: set[Fields], indent: int,
                               welford_output: Welford, timeloop_name: str):

    indent = " " * indent
    indent += "      "

    create_snapshot_string = (indent + "   " + "domainSetup.snapshotHandler_->createSnapshot(timeloop.getCurrentTimeStep(), blocks, {pdf_field}, {force_field}"
                                               "{optional_arguments});")
    if grid == Grid.UNIFORM:
        result = [" [&](){"]
    else:
        result = [" [&](uint_t){"]
    result.append(indent + f"if ((domainSetup.snapshotHandler_->storeSnapshot) && "
                            f"({timeloop_name}.getCurrentTimeStep() % domainSetup.snapshotHandler_->snapshotFrequency == 0) && "
                            f"({timeloop_name}.getCurrentTimeStep() > 0)) {{")

    if target == Target.GPU:
        # field copies
        result.append(indent + "   " + device_to_host_field_copy(Fields.PDF))
        result.append(indent + "   " + device_to_host_field_copy(Fields.FORCE))

        if vtk_output:
            if Fields.MEAN_VELOCITY in field_map:
                result.append(indent + "   " + device_to_host_field_copy(Fields.MEAN_VELOCITY))
            if Fields.SUM_OF_SQUARES in field_map:
                result.append(indent + "   " + device_to_host_field_copy(Fields.SUM_OF_SQUARES))
            if Fields.SUM_OF_CUBES in field_map:
                result.append(indent + "   " + device_to_host_field_copy(Fields.SUM_OF_CUBES))

        result.append(indent + "   gpuDeviceSynchronize();")

    indent_string = indent + " " * len("   domainSetup.snapshotHandler_->createSnapshot(")

    optional_arguments = []
    if Fields.MEAN_VELOCITY in field_map:

        assert (welford_output.mean_field in field_map), \
            "The mean velocity field must be provided in the field map."

        optional_arguments.append(f",\n{indent_string}uint_t({welford_output.sweep_name()}.getCounter())")
        optional_arguments.append(f"{get_field_variable(welford_output.mean_field, Target.CPU)}")

        if len(welford_output.welford_fields) >= 3:
            optional_arguments.append(f"{get_field_variable(welford_output.sos_field, Target.CPU)}")
        if len(welford_output.welford_fields) >= 4:
            optional_arguments.append(f"{get_field_variable(welford_output.soc_field, Target.CPU)}")

    result.append(create_snapshot_string.format(pdf_field=get_field_variable(Fields.PDF, Target.CPU),
                                                force_field=get_field_variable(Fields.FORCE, Target.CPU),
                                                optional_arguments=", ".join(optional_arguments)))

    result.append(indent + "}")
    result.append(indent + "}")

    return "\n".join(result)