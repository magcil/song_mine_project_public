import { Card, Col, Row, Table } from "antd";
import { observer } from "mobx-react";
import { useStores } from "stores/rootStore";
import { columns } from "./columns";
import { useSongs } from "./hooks/useSongs";

const Songs: React.FC = observer(() => {
  const { songsStore } = useStores();
  useSongs();

  return (
    <>
      <div className="tabled">
        <Row gutter={[24, 0]}>
          <Col xs="24" xl={24}>
            <Card
              bordered={false}
              className="criclebox tablespace mb-24"
              title="Songs"
            >
              <div className="table-responsive">
                <Table
                  columns={columns}
                  dataSource={songsStore.songs}
                  className="ant-border-space"
                  pagination={{
                    defaultPageSize: 15,
                    total: songsStore.totalSongs,
                    showSizeChanger: true,
                    onChange: (page, pageSize) => {
                      console.log(page, pageSize);
                      console.log(page, pageSize);
                      songsStore.getAllSongs((page -1 )* (pageSize || 15 ), (pageSize || 15) );
                    },
                  }}
                />
              </div>
            </Card>
          </Col>
        </Row>
      </div>
    </>
  );
});

export default Songs;
