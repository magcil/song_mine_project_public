import { useMemo } from "react";
import { useStores } from "stores/rootStore";

export const useDetections = (
  deviceName?: string,
  page?: number,
  pageSize?: number
) => {
  const { detectionStore } = useStores();

  const detected = useMemo(() => {
    detectionStore.getDetectedData(deviceName, page, pageSize);
    return detectionStore.paginatedResults?.data;
  }, [deviceName, page, pageSize]);

  return { detected };
};
