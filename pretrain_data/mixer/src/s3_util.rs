use std::io;
use std::path::Path;

use aws_sdk_s3::primitives::ByteStream;
use aws_sdk_s3::{Client as S3Client};
use aws_sdk_s3::config::Region;
use tokio::fs::{File as TokioFile};


pub async fn download_to_file(
    s3_client: &S3Client,
    bucket: &str,
    key: &str,
    path: &Path,
) -> Result<(), io::Error> {
    let result = s3_client
        .get_object()
        .bucket(bucket)
        .key(key)
        .send()
        .await
        .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;

    std::fs::create_dir_all(path.parent().unwrap())?;
    let mut file = TokioFile::create(path).await?;
    let mut body = result.body.into_async_read();
    tokio::io::copy(&mut body, &mut file).await?;

    Ok(())
}

pub async fn upload_file(
    s3_client: &S3Client,
    bucket: &str,
    key: &str,
    path: &Path,
) -> Result<(), io::Error> {
    s3_client
        .put_object()
        .bucket(bucket)
        .key(key)
        .body(ByteStream::from_path(path).await?)
        .send()
        .await
        .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;

    Ok(())
}

pub async fn object_size(s3_client: &S3Client, bucket: &str, key: &str) -> Result<usize, io::Error> {
    let resp = s3_client.head_object()
        .bucket(bucket)
        .key(key)
        .send().await
        .map_err(|e| io::Error::new(io::ErrorKind::Other, e));
    match resp {
        Ok(resp) => Ok(resp.content_length as usize),
        Err(e) => Err(e),
    }
}

pub fn find_objects_matching_patterns(s3_client: &S3Client, patterns: &Vec<String>) -> Result<Vec<String>, io::Error> {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build().unwrap();

    let mut stream_inputs: Vec<String> = Vec::new();
    for pattern in patterns.iter() {
        let index = pattern.chars().position(|c| c == '*').unwrap();
        let prefix = pattern[..index].to_string();
        let mut suffix: Option<String> = None;
        if index < pattern.len() - 1 {
            suffix = Some(pattern[index + 2..].to_string());
        }
        let mut has_more = true;
        let mut token: Option<String> = None;
        while has_more {
            let resp =
                if token.is_some() {
                    rt.block_on(s3_client.list_objects_v2()
                        .bucket("ai2-llm")
                        .prefix(&prefix)
                        .delimiter("/")
                        .continuation_token(token.unwrap())
                        .send()).unwrap()
                } else {
                    rt.block_on(s3_client.list_objects_v2()
                        .bucket("ai2-llm")
                        .prefix(&prefix)
                        .delimiter("/")
                        .send()).unwrap()
                };
            resp.contents().unwrap_or_default().iter().for_each(|obj| {
                stream_inputs.push(obj.key().unwrap().to_owned());
            });
            suffix.iter().for_each(|s| {
                resp.common_prefixes().unwrap_or_default().iter().for_each(|sub_folder| {
                    let mut full_path = sub_folder.prefix().unwrap().to_owned();
                    full_path.push_str(s);
                    stream_inputs.push(full_path);
                });
            });
            token = resp.next_continuation_token().map(String::from);
            has_more = token.is_some();
        }
        log::info!("Found {} objects for pattern \"{}\"", stream_inputs.len(), pattern);
    }
    stream_inputs.sort();
    Ok(stream_inputs)
}


pub fn new_client() -> Result<S3Client, io::Error> {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build().unwrap();
    let aws_config = rt.block_on(aws_config::from_env().region(Region::new("us-east-1")).load());
    let s3_client = S3Client::new(&aws_config);
    Ok(s3_client)
}

