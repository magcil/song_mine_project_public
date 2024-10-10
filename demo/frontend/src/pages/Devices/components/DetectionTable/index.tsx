import { Table } from "antd";
import { observer } from "mobx-react";
import { detectionColumns } from "pages/Devices/columns";
import { useDetections } from "pages/Devices/hooks/useDetections";
import { useLayoutEffect, useState } from "react";
import detectionService from "services/museekService/detectionService";
import { ResultDTO } from "services/museekService/detectionService/dtos/detectionModels";

interface IDetectionTableProps {
  name: string;
}
const DetectionTable: React.FC<IDetectionTableProps> = observer(
  ({ name }: IDetectionTableProps) => {
const [data, setData] = useState<ResultDTO[]>([]);

    useLayoutEffect(() => {
        detectionService.getDetected(name).then((res) => {
            setData(res.data);
        });
    }, [name]);
    return <Table columns={detectionColumns} dataSource={data} />;
  }
);

export default DetectionTable;
