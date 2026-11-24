from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import timedelta
from itertools import islice
from typing import Any, Dict, Iterable

import numpy as np

from cognite.client import CogniteClient
from cognite.client.config import global_config
from cognite.client.data_classes.data_modeling import NodeId, ViewId
from cognite.client.data_classes.data_modeling.cdm.v1 import (
    CogniteAsset,
    CogniteTimeSeries,
    CogniteTimeSeriesApply,
)
from cognite.client.data_classes.filters import ContainsAny, Prefix
from cognite.client.exceptions import CogniteNotFoundError


global_config.disable_pypi_version_check = True


def batcher(iterable: Iterable, batch_size: int):
    iterator = iter(iterable)

    while batch := list(islice(iterator, batch_size)):
        yield batch


def get_time_series_for_site(
    client: CogniteClient,
    site: str,
    space: str,
):
    this_site = site.lower()

    sub_tree_root = client.data_modeling.instances.retrieve_nodes(
        NodeId(space, this_site),
        node_cls=CogniteAsset,
    )

    if not sub_tree_root:
        print(
            f"----No CogniteAssets in CDF for {site}!----\n"
            "Run the 'Create Cognite Asset Hierarchy' transformation!"
        )
        return []

    sub_tree_nodes = client.data_modeling.instances.list(
        instance_type=CogniteAsset,
        filter=Prefix(
            property=["cdf_cdm", "CogniteAsset/v1", "path"],
            value=sub_tree_root.path,
        ),
        limit=None,
    )

    if not sub_tree_nodes:
        print(
            f"----No CogniteAssets in the subtree for {site}!----\n"
            "Run the 'Create Cognite Asset Hierarchy' transformation!"
        )
        return []

    value_list = [
        {
            "space": node.space,
            "externalId": node.external_id,
        }
        for node in sub_tree_nodes
    ]

    time_series_batches = [
        client.data_modeling.instances.search(
            view=ViewId(
                "cdf_cdm",
                "CogniteTimeSeries",
                "v1",
            ),
            instance_type=CogniteTimeSeries,
            space=space,
            filter=ContainsAny(
                property=[
                    "cdf_cdm",
                    "CogniteTimeSeries/v1",
                    "assets",
                ],
                values=batch,
            ),
            limit=None,
        )
        for batch in batcher(value_list, 20)
    ]

    # Combine the batch results into a single list.
    time_series = [
        node
        for node_list in time_series_batches
        for node in node_list
    ]

    if not time_series:
        print(
            f"No CogniteTimeSeries found for {site} in the "
            "CogniteCore Data Model."
        )
        return []

    return time_series


def handle(
    client: CogniteClient,
    data: Dict[str, Any] | None = None,
) -> None:
    data = data or {}

    lookback_window = timedelta(
        minutes=data.get("lookback_minutes", 60)
    )

    all_sites = [
        "Houston",
        "Oslo",
        "Kuala_Lumpur",
        "Hannover",
        "Nuremberg",
        "Marseille",
        "Sao_Paulo",
        "Chicago",
        "Rotterdam",
        "London",
    ]

    sites = data.get("sites") or all_sites

    print(f"Processing datapoints for these sites: {sites}")
    print(
        "Lookback window: "
        f"{lookback_window.total_seconds() / 60:g} minutes"
    )

    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_site = {
            executor.submit(
                process_site,
                client,
                lookback_window,
                site,
            ): site
            for site in sites
        }

        for future in as_completed(future_to_site):
            site = future_to_site[future]

            try:
                future.result()
            except Exception as error:
                # Log the failing site while allowing the other sites
                # to finish processing.
                print(
                    f"Failed to process site {site}: "
                    f"{type(error).__name__}: {error}"
                )

                # Re-raise so the Cognite Function call is marked as failed.
                raise


def process_site(
    client: CogniteClient,
    lookback_window: timedelta,
    site: str,
) -> None:
    oee_space = "oee_ts_space"
    source_space = "icapi_dm_space"

    time_series = get_time_series_for_site(
        client=client,
        site=site,
        space=source_space,
    )

    if not time_series:
        print(f"No time series found for {site}. Skipping site.")
        return

    asset_eids = sorted(
        {
            item.external_id.split(":", maxsplit=1)[0]
            for item in time_series
            if item.external_id
        }
    )

    instance_ids = [
        NodeId(
            space=source_space,
            external_id=time_series_node.external_id,
        )
        for time_series_node in time_series
        if time_series_node.external_id
    ]

    if not instance_ids:
        print(f"No valid time-series instance IDs found for {site}.")
        return

    all_latest_dps = client.time_series.data.retrieve_latest(
        instance_id=instance_ids,
    )

    if not all_latest_dps:
        print(f"No latest datapoints found for {site}.")
        return

    # Organize the latest datapoints by equipment.
    # Comparing the equipment part before ":" avoids partial-string
    # matches between equipment external IDs.
    assets_dps = {
        asset_external_id: [
            latest_dp
            for latest_dp in all_latest_dps
            if (
                latest_dp is not None
                and latest_dp.instance_id is not None
                and latest_dp.instance_id.external_id is not None
                and latest_dp.instance_id.external_id.split(
                    ":",
                    maxsplit=1,
                )[0]
                == asset_external_id
            )
        ]
        for asset_external_id in asset_eids
    }

    for asset, latest_dps in assets_dps.items():
        print(f"Calculating OEE for {asset}")

        count_node = f"NodeId({source_space}, {asset}:count)"
        good_node = f"NodeId({source_space}, {asset}:good)"
        status_node = f"NodeId({source_space}, {asset}:status)"
        planned_status_node = (
            f"NodeId({source_space}, {asset}:planned_status)"
        )

        valid_dps = [
            datapoint
            for datapoint in latest_dps
            if (
                datapoint is not None
                and datapoint.instance_id is not None
                and datapoint.timestamp is not None
            )
        ]

        if not valid_dps:
            print(f"{asset}: No valid latest datapoints found.")
            continue

        # timestamp is a datetime in the current SDK.
        # Use the earliest latest timestamp so all required input
        # time series can be aligned to the same complete period.
        end = min(
            datapoint.timestamp
            for datapoint in valid_dps
        )

        # Subtract a timedelta from the datetime timestamp.
        start = end - lookback_window

        input_instance_ids = list(
            {
                datapoint.instance_id
                for datapoint in valid_dps
            }
        )

        if not input_instance_ids:
            print(f"{asset}: No valid input instance IDs found.")
            continue

        print(
            f"{asset}: Retrieving aggregated datapoints "
            f"from {start} to {end}"
        )

        dps_df = client.time_series.data.retrieve_dataframe(
            instance_id=input_instance_ids,
            start=start,
            end=end,
            aggregates=["sum"],
            granularity="1m",
            include_aggregate_name=False,
            limit=None,
        )

        if dps_df is None or dps_df.empty:
            print(
                f"{asset}: No datapoints returned between "
                f"{start} and {end}."
            )
            continue

        required_columns = [
            count_node,
            good_node,
            status_node,
            planned_status_node,
        ]

        missing_columns = [
            column
            for column in required_columns
            if column not in dps_df.columns
        ]

        if missing_columns:
            print(
                f"{asset}: Missing required OEE time-series columns: "
                f"{missing_columns}"
            )
            continue

        # Forward-fill because planned_status and status only receive
        # datapoints when their value changes.
        dps_df = dps_df.ffill()

        # Fill values preceding the first planned_status datapoint
        # with the opposite of the first available state.
        planned_status_first_index = (
            dps_df[planned_status_node].first_valid_index()
        )

        if planned_status_first_index is None:
            print(
                f"{asset}: No datapoints found for "
                f"{planned_status_node}."
            )
            continue

        first_planned_status_value = dps_df.at[
            planned_status_first_index,
            planned_status_node,
        ]

        planned_status_backfill_value = (
            1.0
            if first_planned_status_value == 0.0
            else 0.0
        )

        dps_df[planned_status_node] = (
            dps_df[planned_status_node].fillna(
                value=planned_status_backfill_value
            )
        )

        # Apply the same logic to status.
        status_first_index = dps_df[
            status_node
        ].first_valid_index()

        if status_first_index is None:
            print(
                f"{asset}: No datapoints found for {status_node}."
            )
            continue

        first_status_value = dps_df.at[
            status_first_index,
            status_node,
        ]

        status_backfill_value = (
            1.0
            if first_status_value == 0.0
            else 0.0
        )

        dps_df[status_node] = dps_df[
            status_node
        ].fillna(value=status_backfill_value)

        count_dps = dps_df[count_node]
        good_dps = dps_df[good_node]
        status_dps = dps_df[status_node]
        planned_status_dps = dps_df[planned_status_node]

        total_items = len(count_dps)

        if (
            total_items != len(good_dps)
            or total_items != len(status_dps)
            or total_items != len(planned_status_dps)
        ):
            print(
                f"{asset}: Unable to retrieve aligned datapoints "
                "for all required OEE time series "
                "(count, good, status, planned_status) "
                f"between {start} and {end}."
            )
            continue

        if total_items == 0:
            print(
                f"{asset}: No aligned input datapoints available "
                "for the OEE calculation."
            )
            continue

        off_spec_node = f"{asset}:off_spec"
        quality_node = f"{asset}:quality"
        performance_node = f"{asset}:performance"
        availability_node = f"{asset}:availability"
        oee_node = f"{asset}:oee"

        dps_df[off_spec_node] = count_dps - good_dps

        # Calculate with NumPy while handling zero denominators.
        dps_df[quality_node] = np.divide(
            good_dps,
            count_dps,
            out=np.zeros(
                count_dps.shape,
                dtype=float,
            ),
            where=count_dps.to_numpy() != 0,
        )

        dps_df[performance_node] = np.divide(
            count_dps,
            status_dps,
            out=np.zeros(
                count_dps.shape,
                dtype=float,
            ),
            where=status_dps.to_numpy() != 0,
        ) / (60.0 / 3.0)

        dps_df[availability_node] = np.divide(
            status_dps,
            planned_status_dps,
            out=np.zeros(
                status_dps.shape,
                dtype=float,
            ),
            where=planned_status_dps.to_numpy() != 0,
        )

        dps_df[oee_node] = (
            dps_df[quality_node]
            * dps_df[performance_node]
            * dps_df[availability_node]
        )

        dps_df = dps_df.fillna(value=0.0)
        dps_df = dps_df.replace(
            [np.inf, -np.inf],
            0.0,
        )

        # Keep only calculated OEE output time series.
        dps_df = dps_df.drop(
            columns=[
                count_node,
                good_node,
                status_node,
                planned_status_node,
            ]
        )

        to_insert = [
            {
                "instance_id": NodeId(
                    space=oee_space,
                    external_id=external_id,
                ),
                "datapoints": [
                    (
                        timestamp,
                        float(value),
                    )
                    for timestamp, value in zip(
                        dps_df.index,
                        dps_df[external_id].to_numpy(),
                    )
                ],
            }
            for external_id in dps_df.columns
        ]

        if not to_insert:
            print(f"{asset}: No calculated datapoints to insert.")
            continue

        try:
            client.time_series.data.insert_multiple(
                to_insert
            )

        except CogniteNotFoundError as error:
            # Create missing OEE time-series nodes and retry insertion.
            time_series_to_create = []

            for missing_item in error.not_found:
                instance_id = missing_item.get("instanceId", {})
                external_id = instance_id.get("externalId")

                if not external_id:
                    print(
                        f"{asset}: Unable to determine missing "
                        f"time-series external ID from {missing_item}."
                    )
                    continue

                print(
                    f"Creating CogniteTimeSeries "
                    f"{oee_space}/{external_id}"
                )

                name_parts = external_id.split(":")
                name_parts[-1] = (
                    name_parts[-1]
                    .replace("_", " ")
                    .title()
                )

                time_series_to_create.append(
                    CogniteTimeSeriesApply(
                        space=oee_space,
                        external_id=external_id,
                        name=" ".join(name_parts),
                        is_step=False,
                        time_series_type="numeric",
                    )
                )

            if not time_series_to_create:
                raise

            client.data_modeling.instances.apply(
                time_series_to_create
            )

            client.time_series.data.insert_multiple(
                to_insert
            )

        print(
            f"{asset}: Inserted {len(dps_df)} datapoints "
            f"for {len(dps_df.columns)} OEE time series."
        )